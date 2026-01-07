import logging
from method.RAG import RAG
from method.QAModels import QAModel
from method.EmbeddingModels import SnowflakeArcticEmbeddingModel
from utils import *
from datetime import datetime
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")
FILE_DIR = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews"
META_DIR = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews_meta"
STORED_NODES_PATH = f"./dos_rag/save/nodes"
LOG_DIR = "./dos_rag/save/logs/"
LOG_FILE = f"{LOG_DIR}/run_rag_{CURRENT_DATE_TIME}.log"
MODEL_NAME = "Qwen/Qwen3-4B"
ANSWER_PATH = "./dos_rag/save/answer"

create_directories([ANSWER_PATH, LOG_DIR, STORED_NODES_PATH])


def run(filename, qa_model, embedding_model, hyper_param):
    meta_data_dict = defaultdict(dict)
    with open(os.path.join(FILE_DIR, filename), "r") as f:
        reviews = json.load(f)
    with open(os.path.join(META_DIR, filename), "r") as f:
        for line in f:
            data = json.loads(line)
            meta_data_dict[data["parent_asin"]] = data

    rag = RAG(chunk_size=100, embedding_model=embedding_model, qa_model=qa_model)
    rag.load_nodes(os.path.join(STORED_NODES_PATH, filename))

    pairs = []
    for review in reviews[-10:]:
        meta_data = meta_data_dict[review["parent_asin"]]
        answer, context, node_information = rag.answer_question(
            meta_data, 
            top_k=hyper_param["top_k"], 
            max_tokens=hyper_param["max_tokens"]
        )
        answer = extract_review(answer)
        pairs.append({"model":answer, "people":review["text"]})
    
    with open(os.path.join(ANSWER_PATH, f"{filename}.json"), "w") as fp:
        json.dump(pairs, fp, indent=2)

    
def run_experiment(hyper_param):
    logger = create_logger(LOG_FILE)
    qa_model = QAModel(MODEL_NAME)
    embedding_model = SnowflakeArcticEmbeddingModel()
    file_list = os.listdir(META_DIR)
    for filename in file_list:
        run(filename, qa_model, embedding_model, hyper_param)


if __name__ == '__main__':
    hyper_param = {"top_k":10, "max_tokens":2048}
    run_experiment(hyper_param)