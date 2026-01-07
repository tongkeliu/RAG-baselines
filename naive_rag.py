from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
import torch
import os
import json
import re
from benchmarks import *

# construct pipeline
model_name = "Qwen/Qwen3-4B"
# model_name = "Qwen/Qwen3-14B-AWQ"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True)
pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer,
    max_length=2048,
    eos_token_id=tokenizer.eos_token_id, 
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False  # set True to check the full prompt
)
llm = HuggingFacePipeline(pipeline=pipe)

# read sampled single user reviews (about 200 reviews)
data_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews"
filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
with open(os.path.join(data_dir, filename), "r") as f:
    selected_reviews = json.load(f)

# construct langchain elements
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = [Document(page_content=str(doc['text'])) for doc in selected_reviews[:-10]] # 10 test data
splits = text_splitter.split_documents(docs)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt_template = """<|im_start|>user
Based on the following past reviews from user AE22QOZDLAODPKJHYZI2HXQ7MLZQ and the book information, write a new Amazon book review in their exact voice and style. The review must be 40-80 words long. Do not include any explanations, introductions, or meta-commentary. Only output the review text. 
Enclose your entire output strictly between [[[ and ]]] like this:
[[[Your review text here.]]]

Past reviews:
{context}

Book: {book_title}
Author: {book_author}
Summary: {book_summary}

Review: <|im_end|>
<|im_start|>assistant /no_think
"""

def format_docs(docs):
    formatted = "\n\n".join(doc.page_content for doc in docs)
    return formatted

def rag_generate(book_title, book_author, book_summary):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "book_title": RunnableLambda(lambda _: book_title),
            "book_author": RunnableLambda(lambda _: book_author),
            "book_summary": RunnableLambda(lambda _: book_summary),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(f"title:{book_title} author:{book_author} summary:{book_summary}").strip()

def extract_review(output):
    match = re.search(r"\[\[\[(.*?)\]\]\]", output, re.DOTALL)
    if match:
        review = match.group(1).strip()
    else:
        review = "I don't know"
    return review

def find_meta_data(parent_asin, fp):
    for line in fp:
        meta = json.loads(line)
        if meta['parent_asin'] == parent_asin:
            return meta
    return None

# load book meta data
meta_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews_meta"
filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
meteor_list = []
rouge_list = [[], [], []]
ppl_list = []

with open(os.path.join(meta_dir, filename), "r") as fin:
    for review in selected_reviews[-10:]:
        parent_asin = review['parent_asin']
        meta = find_meta_data(parent_asin, fin)
        if meta == None:
            break
        book_title = meta['title']
        book_author = meta['author']['name']
        book_summary = meta['book_summary']
        output = rag_generate(book_title, book_author, book_summary)
        final_review = extract_review(output)
        # print(final_review)    

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