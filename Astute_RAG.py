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


def load_llm(model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer,
        max_length=20000,
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False  # set True to check the full prompt
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


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


def create_retrieval_chain(vectorstore, llm, book_title, book_author, book_summary):
    prompt_template = """
    Based on the following past reviews from user AE22QOZDLAODPKJHYZI2HXQ7MLZQ and the book information, write a new Amazon book review in their exact voice and style. The review must be 40-80 words long. Do not include any explanations, introductions, or meta-commentary. Only output the review text. 
    Enclose your entire output strictly between [[[ and ]]] like this:
    [[[Your review text here.]]]

    Past reviews:
    {context}

    Book: {book_title}
    Author: {book_author}
    Summary: {book_summary}

    Review:"""
    prompt_template = apply_chat_template(prompt_template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            # 3. 这里只为生成答案准备输入，把 raw docs 转换成字符串
            context=(lambda x: format_docs(x["context"])),
            book_title=lambda _: book_title,
            book_author=lambda _: book_author,
            book_summary=lambda _: book_summary,
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {
            "context": retriever, 
            "question": RunnablePassthrough() 
        }
    ).assign(
        answer=rag_chain_from_docs 
    )
    return rag_chain_with_source


# Adaptive Internal Knowledge Generation
def generate_internal_knowledge(query, llm, max_passages=1):
    prompt = f"""Generate a document that provides accurate and relevant information to answer the given question.
    If the information is unclear or uncertain, explicitly state "I don't know" and then stop to avoid any hallucinations.
    Question: {query} Document:"""
    res = []
    for i in range(max_passages):
        response = llm.generate([apply_chat_template(prompt)], max_length=4096)
        text = response.generations[0][0].text.strip("<think>\n\n</think>\n\n")
        res.append({"source": "internal", "text": text})
    return res


# Astute RAG Implementation
def astute_rag(query, rag_chain, llm, t, meta_data):
    # Step 1: Retrieve External Passages
    retrieval_results = rag_chain.invoke(query)
    external_passages = [{"source": "external", "text": doc.page_content} for doc in retrieval_results["context"]]

    # Step 2: Generate Internal Knowledge
    internal_passages = generate_internal_knowledge(query, llm)

    # Step 3: Consolidate Knowledge
    all_passages = internal_passages + external_passages
    # sources = [p["source"] for p in all_passages]
    # texts = [p["text"] for p in all_passages]
    last_context = initial_context = all_passages
    
    for i in range(t):
        consolidation_prompt = f"""Task: Consolidate information from both your own memorized documents and externally retrieved documents in response to the given question.
        * For documents that provide consistent information, cluster them together and summarize the key details into a single, concise document.
        * For documents with conflicting information, separate them into distinct documents, ensuring each captures the unique perspective or data.
        * Exclude any information irrelevant to the query.
        For each new document created, clearly indicate:
        * Whether the source was from memory or an external retrieval.
        * The original document numbers for transparency.
        Initial Context: {str(initial_context)}
        Last Context: {str(last_context)}
        Question: {query}
        New Context:"""
        consolidation_prompt = apply_chat_template(consolidation_prompt)
        consolidated_response = llm.generate([consolidation_prompt], max_length=4096).generations[0][0].text
        last_context = consolidated_response.strip("<think>\n\n</think>\n\n")

    # Step 4: Finalize Answer
    final_prompt = f"""Task: Answer a given question using the consolidated information from both your own memorized documents and externally retrieved documents.
    Step 1: Consolidate information
    * For documents that provide consistent information, cluster them together and summarize the key details into a single, concise document.
    * For documents with conflicting information, separate them into distinct documents, ensuring each captures the unique perspective or data.
    * Exclude any information irrelevant to the query.
    For each new document created, clearly indicate:
    * Whether the source was from memory or an external retrieval.
    * The original document numbers for transparency.
    Step 2: Propose Answers and Assign Confidence
    For each group of documents, propose a possible answer and assign a confidence score based on the credibility and agreement of the information.
    Step 3: Select the Final Answer
    After evaluating all groups, select the most accurate and well-supported answer.
    Highlight your exact answer within [[[your answer]]].

    Based on the following past reviews from user AE22QOZDLAODPKJHYZI2HXQ7MLZQ and the book information, write a new Amazon book review in their exact voice and style. The review must be 40-80 words long. Do not include any explanations, introductions, or meta-commentary. Only output the review text. 
    Enclose your entire output strictly between [[[ and ]]] like this:
    [[[Your review text here.]]]

    Book: {meta_data[0]}
    Author: {meta_data[1]}
    Summary: {meta_data[2]}

    Initial Context (user's past reviews): {initial_context}
    Consolidated Context (user's past reviews): {last_context}
    Question: {query}
    Answer:"""
    final_prompt = apply_chat_template(final_prompt)
    final_answer = llm.generate([final_prompt], max_length=4096).generations[0][0].text

    return final_answer.strip("<think>\n\n</think>\n\n")


def extract_review(output):
    match = re.search(r"\[\[\[(.*?)\]\]\]", output, re.DOTALL)
    if match:
        review = match.group(1).strip()
    else:
        review = "I don't know"
    return review


def main():
    data_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews"
    filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
    model_name = "Qwen/Qwen3-4B"
    iterations = 2
    documents = load_documents_from_folder(data_dir, filename)
    vectorstore = create_vectorstore(documents)
    llm = load_llm(model_name)

    meteor_list = []
    rouge_list = [[], [], []]
    ppl_list = []
    meta_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews_meta"
    filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
    astute_rag_output = "./astute_rag.json"
    res_dict = defaultdict(dict)
    for review in documents[-5:]:
        parent_asin = review['parent_asin']
        meta_data = load_meta_data_from_folder(meta_dir, filename, parent_asin)

        query = "Write a book review in the user's voice."
        rag_chain = create_retrieval_chain(vectorstore, llm, *meta_data)
        result = astute_rag(query, rag_chain, llm, iterations, meta_data)
        final_review = extract_review(result)
        res_dict[parent_asin] = (final_review, review['text'])
        # print("Final Answer:", result)

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
    with open(astute_rag_output, "w") as fp:
        json.dump(res_dict, fp, indent=2)


if __name__ == "__main__":
    main()