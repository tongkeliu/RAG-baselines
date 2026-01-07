from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
import torch
import os
import json
import re
from benchmarks import *
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np

model_name = "Qwen/Qwen3-4B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True)

def answer_question(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content


def load_documents_from_folder(data_dir, filename):
    with open(os.path.join(data_dir, filename), "r") as f:
        selected_reviews = json.load(f)
    return selected_reviews


def load_meta_data_from_folder(data_dir, filename, parent_asin):
    with open(os.path.join(data_dir, filename), "r") as fin:
        for line in fin:
            meta = json.loads(line)
            if meta['parent_asin'] == parent_asin:
                break
        book_title = meta['title']
        book_author = meta['author']['name']
        book_summary = meta['book_summary']
        return book_title, book_author, book_summary


# Embed Documents and Create a VectorStore
def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = [Document(page_content=str(doc['text'])) for doc in documents[:-10]] # 10 test data
    splits = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embedding_model)

    return vectorstore


# Create a Retrieval-based QA Chain
def format_docs(docs):
    formatted = "\n\n".join(doc.page_content for doc in docs)
    return formatted


def apply_chat_template(prompt):
    return f"""<|im_start|>user
    {prompt} <|im_end|>
    <|im_start|>assistant /no_think
    """


def extract_review(output):
    match = re.search(r"\[\[\[(.*?)\]\]\]", output, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = "I don't know"

    match = re.search(r"\(\(\((.*?)\)\)\)", output, re.DOTALL)
    if match:
        rationale = match.group(1).strip()
    else:
        rationale = "I don't know" 
    return answer, rationale


def drafter(vectorstore, book_title, book_author, book_summary):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(f"title:{book_title} author:{book_author} summary:{book_summary}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])

    kmeans = KMeans(n_clusters=4)
    clusters = kmeans.fit_predict(X=embeddings)
    unique_clusters = set(clusters)
    cluster_dict = defaultdict(list)
    for index, cluster in enumerate(clusters):
        cluster_dict[cluster].append(index)
    m = min(len(indices) for indices in cluster_dict.values())

    subsets: list[list[str]] = []
    for _ in range(m):
        subset: list[int] = []
        for cluster in unique_clusters:
            chosen_element: int = np.random.choice(cluster_dict[cluster])
            subset.append(chosen_element)
            cluster_dict[cluster].remove(chosen_element)

        subset_documents = [docs[idx].page_content for idx in subset]
        subsets.append(subset_documents)
    
    instruction = """Response to the instruction. Also provide rationale for your response.
    ## Instruction: Based on the following past reviews from user AE22QOZDLAODPKJHYZI2HXQ7MLZQ and the book information, write a new Amazon book review in their exact voice and style. The review must be 40-80 words long.
    Enclose your output strictly between [[[ and ]]] like this:
    [[[Your review text here.]]]
    Enclose the rationale for your response between ((( and ))) like this
    (((Your rationale text here.)))

    Past reviews:
    {context}

    Book: {book_title}
    Author: {book_author}
    Summary: {book_summary}

    Response:"""

    drafter_answers = []
    for subset in subsets:
        formated_instruction = instruction.format(context=subset, book_title=book_title, book_author=book_author, book_summary=book_summary)
        response = answer_question(formated_instruction)
        answer, rationale = extract_review(response)
        drafter_answers.append((answer, rationale))
    return drafter_answers


def verifier(drafter_answers):
    instruction = """Based on the following past reviews from user AE22QOZDLAODPKJHYZI2HXQ7MLZQ and the book information, write a new Amazon book review in their exact voice and style. The review must be 40-80 words long."""
    rag_verifier_prompt: str = """
    ## Instruction: {instruction}

    ## Response: {response} 

    ## Rationale: {rationale}

    Is the rationale good enough to support the answer? (Yes or No)"""

    verified_answers = []
    for answer, rationale in drafter_answers:
        response = answer_question(rag_verifier_prompt.format(instruction=instruction, response=answer, rationale=rationale))
        if "yes" in response.lower():
            verified_answers.append(answer)
    return np.random.choice(verified_answers)


def main():
    data_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews"
    filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
    documents = load_documents_from_folder(data_dir, filename)
    vectorstore = create_vectorstore(documents)

    meteor_list = []
    rouge_list = [[], [], []]
    ppl_list = []
    meta_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews_meta"
    filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
    rag_output = "./speculative_rag.json"
    res_dict = defaultdict(dict)
    for review in documents[-5:]:
        parent_asin = review['parent_asin']
        meta_data = load_meta_data_from_folder(meta_dir, filename, parent_asin)
        drafter_answers = drafter(vectorstore, *meta_data)
        final_review = verifier(drafter_answers)
        res_dict[parent_asin] = (final_review, review['text'])
        # print("Final Answer:", final_review)

        meteor_list.append(compute_meteor(final_review, review['text']))
        rouge = compute_rouge(final_review, review['text'])
        rouge_list[0].append(rouge["rouge1"])
        rouge_list[1].append(rouge["rouge2"])
        rouge_list[2].append(rouge["rougeL"])
        ppl_list.append(compute_ppl(final_review))

    def avg(input: list):
        return sum(input) / len(input)

    print(f"METEOR: {avg(meteor_list):.4f}")
    print(f"ROUGE-1: {avg(rouge_list[0]):.4f}")
    print(f"ROUGE-2: {avg(rouge_list[1]):.4f}")
    print(f"ROUGE-L: {avg(rouge_list[2]):.4f}")
    print(f"Perplexity (PPL): {avg(ppl_list):.2f}")
    with open(rag_output, "w") as fp:
        json.dump(res_dict, fp, indent=2)


if __name__ == "__main__":
    main()