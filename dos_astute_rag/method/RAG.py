import tiktoken
import pickle
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance 
from typing import Dict, List, Set
import numpy as np
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel
from .utils import split_text

class Node:
    """
    Represents a node in the RAG system.
    """
    def __init__(self, text: str, index: int, embedding) -> None:
        self.text = text
        self.index = index
        self.embedding = embedding


class RAG:
    """
    A class for managing the Retrieval-Augmented Generation (RAG) process.
    """
    def __init__(self, chunk_size: int, embedding_model: BaseEmbeddingModel, qa_model: BaseQAModel):
        """
        Initializes the RAG instance with an embedding model and tokenizer.

        Parameters:
            max_tokens (int): Maximum number of tokens per chunk.
            embedding_model (BaseEmbeddingModel): Model used to create query and chunk embeddings.
        """
        self.qa_model = qa_model
        self.embedding_model = embedding_model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = chunk_size
        self.nodes = {}


    def create_node(self, index: int, text: str):
        """
        Creates a Node object for a given text chunk.

        Parameters:
            index (int): The index of the chunk.
            text (str): The text content of the chunk.
        """
        embedding = self.embedding_model.create_embedding(text)
        return Node(text, index, embedding)


    def chunk_and_embed_document(self, text: str):
        """
        Splits a document into chunks and embeds each chunk.

        Parameters:
            text (str): The document to process.

        Returns:
            dict: A dictionary of Node objects indexed by chunk index.
        """
        chunks = split_text(text, self.tokenizer, self.chunk_size)
        nodes = {}
        for index, chunk in enumerate(chunks):
            node = self.create_node(index, chunk)
            nodes[index] = node

        self.nodes = nodes
        
        return nodes


    def store_nodes(self, path):
        if self.nodes is None:
            raise ValueError("There are no nodes.")
        with open(path, "wb") as file:
            pickle.dump(self.nodes, file)


    def load_nodes(self, path):
        """
        Load stored nodes from a specified path into the RAG object.

        Parameters:
            path (str): Path to the file where nodes are stored.

        Raises:
            ValueError: If the file cannot be loaded or if the loaded object is invalid.
        """
        if not isinstance(path, str):
            raise ValueError("The path must be a string.")

        try:
            with open(path, "rb") as file:
                loaded_nodes = pickle.load(file)
            if not isinstance(loaded_nodes, dict):
                raise ValueError("The loaded object is not a dictionary of nodes.")
            for key, value in loaded_nodes.items():
                if not isinstance(value, Node):
                    raise ValueError(f"Node at key {key} is not an instance of Node.")
            self.nodes = loaded_nodes
        except Exception as e:
            raise ValueError(f"Failed to load nodes from {path}: {e}")


    # Adaptive Internal Knowledge Generation
    def generate_internal_knowledge(self, query, max_passages=1):
        prompt = f"""Generate a document that provides accurate and relevant information to answer the given question.
        If the information is unclear or uncertain, explicitly state "I don't know" and then stop to avoid any hallucinations.
        Question: {query} Document:"""
        res = []
        for i in range(max_passages):
            response = self.qa_model.generate(prompt)
            res.append({"source": "internal", "text": response})
        return res


    # Astute RAG Implementation
    def astute_rag(self, query, t, retrieval_results):
        # Step 1: Retrieve External Passages
        external_passages = [{"source":"external", "text":result} for result in retrieval_results]

        # Step 2: Generate Internal Knowledge
        internal_passages = self.generate_internal_knowledge(query)

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
            consolidated_response = self.qa_model.generate(consolidation_prompt)
            last_context = consolidated_response
        
        return last_context

        # # Step 4: Finalize Answer
        # final_prompt = f"""Task: Answer a given question using the consolidated information from both your own memorized documents and externally retrieved documents.
        # Step 1: Consolidate information
        # * For documents that provide consistent information, cluster them together and summarize the key details into a single, concise document.
        # * For documents with conflicting information, separate them into distinct documents, ensuring each captures the unique perspective or data.
        # * Exclude any information irrelevant to the query.
        # For each new document created, clearly indicate:
        # * Whether the source was from memory or an external retrieval.
        # * The original document numbers for transparency.
        # Step 2: Propose Answers and Assign Confidence
        # For each group of documents, propose a possible answer and assign a confidence score based on the credibility and agreement of the information.
        # Step 3: Select the Final Answer
        # After evaluating all groups, select the most accurate and well-supported answer.
        # Highlight your exact answer within [[[your answer]]].
        # Initial Context: {initial_context}
        # [Consolidated Context: {last_context}]
        # Question: {query}
        # Answer:"""
        # final_answer = self.qa_model.generate(final_prompt)

        # return final_answer

    def answer_question(self,
        meta_data = None, #in case of multiple-choice
        top_k: int = 15,
        max_tokens: int = 1500,
        ):


        context, node_information = self.retrieve(
            query=str(meta_data), top_k=top_k, max_tokens=max_tokens
        )
        query = "Write a book review in the user's voice."
        last_context = self.astute_rag(query, 0, context)


        #for multiple choice questions
        answer = self.qa_model.answer_question(last_context, meta_data)

        return answer, context, node_information

    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        max_tokens: int = 3500
    ) -> str:
        """
        Retrieve the most relevant nodes for a given query based on cosine similarity.

        Parameters:
            query (str): The input query string.
            top_k (int): Maximum number of nodes to retrieve.
            max_tokens (int): Maximum token limit for the concatenated context.

        Returns:
            tuple: A tuple containing the concatenated context (str) and a list of retrieved node IDs (list[int]).
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty or only whitespace")

        if not self.nodes:
            raise ValueError("No nodes available for retrieval.")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")


        # Step 1: Compute the query embedding
        query_embedding = self.embedding_model.create_query_embedding(query)

        # Step 2: Load all node embeddings as list
        node_list = sorted(self.nodes.values(), key=lambda x: x.index)
        embeddings = [node.embedding for node in node_list]

        # Step 3: Compute cosine similarity scores
        distances = [
            distance.cosine(query_embedding, embedding)
            for embedding in embeddings
        ]
        sorted_indices = np.argsort(distances)

        # Step 4: Select top-k nodes within the token limit
        selected_nodes = []
        total_tokens = 0
        for index in sorted_indices[:top_k]:
            node = node_list[index]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        # Step 5: Concatenate selected node texts
        sorted_node_list = sorted(selected_nodes, key=lambda node: node.index)
        context = " \n\n".join(node.text.replace("\n", " ") for node in sorted_node_list)

        # Step 6: Return the context and node IDs
        retrieved_node_ids = [node.index for node in selected_nodes]

        return [node.text for node in sorted_node_list], retrieved_node_ids