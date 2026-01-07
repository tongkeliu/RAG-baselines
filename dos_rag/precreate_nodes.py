import logging
from method.RAG import RAG
from method.QAModels import QAModel
from method.EmbeddingModels import SnowflakeArcticEmbeddingModel
from utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file, openFileWithUnknownEncoding, count_words, create_logger
from datetime import datetime
import os
import json


# Paths
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")
FILE_DIR = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews"
STORED_NODES_PATH = f"./dos_rag/save/nodes"
LOG_DIR = "./dos_rag/save/logs/"
LOG_FILE = f"{LOG_DIR}/{CURRENT_DATE_TIME}.log"
MODEL_NAME = "Qwen/Qwen3-4B"

# Ensure necessary directories exist
create_directories([STORED_NODES_PATH, LOG_DIR])
logger = create_logger(LOG_FILE)

# Hyperparameters
HYPERPARAMS = {
    "chunk_size": 500,  # Chunk size for splitting documents
}


def precreate_nodes():
    logging.info("Starting precreate_nodes process...")
    # Initialize models
    logging.info("Initializing models...")

    qa_model = QAModel(model_name=MODEL_NAME)
    embedding_model = SnowflakeArcticEmbeddingModel()

    # Initialize RAG pipeline
    rag = RAG(chunk_size=HYPERPARAMS["chunk_size"], embedding_model=embedding_model, qa_model=qa_model)
    file_list = os.listdir(FILE_DIR)
    for filename in file_list:
        try:
            if filename != "AE22QOZDLAODPKJHYZI2HXQ7MLZQ":
                continue
            logging.info(f"Processing document {filename}...")
            with open(os.path.join(FILE_DIR, filename), "r") as f:
                reviews = json.load(f)
                review_text = "\n\n".join(review['text'] for review in reviews[:-10])
                rag.chunk_and_embed_document(review_text)
                logging.info(f"Successfully chunked and embedded document {filename}.")

                nodes_file_path = f"{STORED_NODES_PATH}/{filename}"
                rag.store_nodes(nodes_file_path)
                logging.info(f"Saved nodes for document {filename} at {nodes_file_path}.")
        
        except Exception as e:
            logging.exception(f"Error processing document_id {filename}: {e}")
            log_error(filename, "-", str(e), f"{STORED_NODES_PATH}/error_{filename}")

if __name__ == "__main__":
    precreate_nodes()