import argparse
import numpy as np
import json
import os
import multiprocessing
from abc import ABC, abstractmethod
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict
from utils.prompt import detailed_table_ranking_prompt
from utils.gpt import OpenAIClient
from utils.groq import GroqClient
from utils.pickle_utils import load_pickle_file
from utils.table_utils import (
    generate_tables_random_rows,
    generate_tables_rel_rows,
    generate_tables_with_schema,
    build_row_corpus,
    convert_table_contents_to_dfs,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt

load_dotenv()

# Global variable for the row database.
row_db = None

def process_single_query_worker(query_idx: int, query_embedding, retrieved_ids, table_names, num_candidates):
    """
    Worker function to process a single query.
    Uses the globally inherited `row_db` (loaded in main via fork) to perform similarity search.
    """
    global row_db
    total_docs = row_db.index.ntotal  # Total number of documents in the index.
    docs = row_db.similarity_search_by_vector(query_embedding, k=total_docs)
    
    # Organize documents by table index.
    table_docs = defaultdict(list)
    for doc in docs:
        table_docs[doc.metadata["table_index"]].append(doc.page_content)
        
    candidate_table = generate_tables_rel_rows(retrieved_ids, table_docs, table_names, num_candidates)
    return query_idx, candidate_table

class BaseReranker(ABC):
    """Base class for re-ranking retrieved tables."""
    
    def __init__(self, model_name: str, dataset_name: str, ranking_method: str, output_dir: str, eval_dir: str, num_candidates: int):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ranking_method = ranking_method
        self.output_dir = output_dir
        self.eval_dir = eval_dir
        self.client = OpenAIClient() if model_name == "openai" else GroqClient(model=model_name)
        self.retrieval_dir = "./retriever/outputs"
        self.num_candidates = num_candidates
        self.ranked_ids_path = os.path.join(
            self.output_dir, f"{self.model_name}_{self.dataset_name}_{self.ranking_method}_{self.num_candidates}c_reranked_ids.npy"
        )
        # Create output directories if they don't exist.
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Load dataset and retrievals.
        self.load_data()
        
    def load_data(self) -> None:
        """Load the processed dataset and retrieval results."""
        dataset_dir = f"./data/{self.dataset_name}"
        
        # Load dataset files.
        self.table_contents = load_pickle_file(f"{dataset_dir}/table_contents.pkl")
        self.table_names = load_pickle_file(f"{dataset_dir}/table_names.pkl")
        self.validation_data = load_pickle_file(f"{dataset_dir}/validation_data.pkl")
        
        # Load retrieved IDs.
        self.retrieved_ids = np.load(
            f"{self.retrieval_dir}/openai_{self.dataset_name}_retrieved_ids.npy"
        ).tolist()

    def get_questions(self) -> List:
        """Get list of questions from validation data."""
        return [val_data['question'] for val_data in self.validation_data]

    @abstractmethod
    def get_candidate_tables(self, **kwargs) -> List[str]:
        """Get candidate tables for re-ranking."""
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def rerank_and_eval(self) -> Dict[str, float]:
        raise NotImplementedError("Subclass must implement this method")

    def create_prompts(self, questions: List[str], candidate_tables: List[str]) -> List[str]:
        """Create prompts for re-ranking."""
        return [
            detailed_table_ranking_prompt.format(
                question=questions[i], 
                candidate_tables=candidate_tables[i].rstrip()
            )
            for i in range(len(questions))
        ]
    
    def rerank_all(self, prompts: List[str]) -> np.ndarray:
        """Re-rank tables for all queries."""
        if self.dataset_name == "spider":
            min_id, max_id = 0, 728
        elif self.dataset_name == "atis":
            min_id, max_id = 0, 11
        elif self.dataset_name == "geoq":
            min_id, max_id = 0, 7
        fallback_ids = [ids[:5] for ids in self.retrieved_ids]
        reranked_ids = self.client.process_prompts(prompts, min_val=min_id, max_val=max_id, fallback_ids=fallback_ids)
        # Convert the list of table_ids to numpy array.
        reranked_ids = np.array(reranked_ids)
        # Save the reranked_ids.
        np.save(self.ranked_ids_path, reranked_ids)
        return reranked_ids
    
    def evaluate(self, reranked_ids) -> Dict[str, Dict[str, float]]:
        """
        Evaluate re-ranking performance and break down results by table count.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing evaluation metrics per table count.
        """
        queries_by_table_count = defaultdict(list)
        for i, query_data in enumerate(self.validation_data):
            num_tables = len(query_data['table_ids'])
            if num_tables > 0:
                queries_by_table_count[num_tables].append(i)
                
        metrics_by_table_count = defaultdict(lambda: defaultdict(list))
        overall_metrics = defaultdict(list)
        k = 5  # Fixed k for evaluation.
        
        for num_tables, query_indices in sorted(queries_by_table_count.items(), key=lambda x: x[0]):
            for i in query_indices:
                query_data = self.validation_data[i]
                gold_table_ids = set(query_data['table_ids'])
                if not gold_table_ids:
                    continue
                
                pred_table_ids = set(reranked_ids[i][:k])
                n_correct = len(gold_table_ids.intersection(pred_table_ids))
                
                recall = n_correct / len(gold_table_ids)
                precision = n_correct / k if k > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics_by_table_count[f"{num_tables}-table(s)"]['precision'].append(precision)
                metrics_by_table_count[f"{num_tables}-table(s)"]['recall'].append(recall)
                metrics_by_table_count[f"{num_tables}-table(s)"]['f1'].append(f1)
                
                overall_metrics['precision'].append(precision)
                overall_metrics['recall'].append(recall)
                overall_metrics['f1'].append(f1)
        
        results = {}
        for num_tables, metrics in metrics_by_table_count.items():
            results[f"{num_tables}"] = {
                'precision@5': np.mean(metrics['precision']),
                'recall@5': np.mean(metrics['recall']),
                'f1@5': np.mean(metrics['f1'])
            }
        
        results['overall'] = {
            'precision@5': np.mean(overall_metrics['precision']),
            'recall@5': np.mean(overall_metrics['recall']),
            'f1@5': np.mean(overall_metrics['f1'])
        }
        
        results_path = os.path.join(
            self.eval_dir, 
            f"{self.model_name}_{self.dataset_name}_{self.ranking_method}_{self.num_candidates}_rerank_results.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nRe-ranking using {self.ranking_method} method with {self.num_candidates} tables:")
        print(json.dumps(results, indent=2))
        
        return results

class SchemaReranker(BaseReranker):
    """Re-ranker using only schema information."""
    
    def get_candidate_tables(self) -> List[str]:
        return generate_tables_with_schema(self.retrieved_ids, self.table_names, self.table_contents, k=self.num_candidates)

    def rerank_and_eval(self) -> Dict[str, float]:
        if os.path.exists(self.ranked_ids_path):
            return self.evaluate(np.load(self.ranked_ids_path))
        candidate_tables = self.get_candidate_tables()
        questions = self.get_questions()
        reranked_ids = self.rerank_all(self.create_prompts(questions, candidate_tables))
        return self.evaluate(reranked_ids)

class RandomRowsReranker(BaseReranker):
    def get_candidate_tables(self) -> List[str]:
        return generate_tables_random_rows(self.retrieved_ids, self.table_names, self.table_contents, k=self.num_candidates)

    def rerank_and_eval(self) -> Dict[str, float]:
        if os.path.exists(self.ranked_ids_path):
            return self.evaluate(np.load(self.ranked_ids_path))
        candidate_tables = self.get_candidate_tables()
        questions = self.get_questions()
        prompts = self.create_prompts(questions, candidate_tables)
        reranked_ids = self.rerank_all(prompts)
        return self.evaluate(reranked_ids)

class RelevantRowsReranker(BaseReranker):
    """Re-ranker using relevant rows."""
    
    def __init__(self, model_name: str, dataset_name: str, ranking_method: str, output_dir: str, eval_dir: str, num_candidates: int):
        super().__init__(model_name, dataset_name, ranking_method, output_dir, eval_dir, num_candidates)
        self.query_embeddings = np.load(
            f"{self.retrieval_dir}/openai_{self.dataset_name}_query_embeddings.npy"
        )

    def get_candidate_tables(self) -> List[str]:
        total_queries = len(self.get_questions())
        candidate_tables = [None] * total_queries

        # Use ProcessPoolExecutor with the "fork" context so children inherit the global row_db.
        # (macOS/Windows default to "spawn", which would leave row_db as None in workers.)
        fork_context = multiprocessing.get_context("fork")
        with tqdm(total=total_queries, desc="Processing queries") as progress_bar:
            with ProcessPoolExecutor(max_workers=8, mp_context=fork_context) as executor:
                futures = []
                for idx in range(total_queries):
                    futures.append(
                        executor.submit(
                            process_single_query_worker,
                            idx,
                            self.query_embeddings[idx],
                            self.retrieved_ids[idx],
                            self.table_names,
                            self.num_candidates
                        )
                    )
                for future in as_completed(futures):
                    idx, candidate_table = future.result()
                    candidate_tables[idx] = candidate_table
                    progress_bar.update(1)
        
        return candidate_tables

    def rerank_and_eval(self) -> Dict[str, float]:
        if os.path.exists(self.ranked_ids_path):
            return self.evaluate(np.load(self.ranked_ids_path))
        candidate_tables = self.get_candidate_tables()
        questions = self.get_questions()
        prompts = self.create_prompts(questions, candidate_tables)
        reranked_ids = self.rerank_all(prompts)
        return self.evaluate(reranked_ids)

    def evaluate_with_file(self, reranked_ids) -> Dict[str, Dict[str, float]]:
        # load pickle file
        dic = sorted(load_pickle_file(reranked_ids).items())
        ranked_list = [[] for _ in range(985)]
        for item in dic:
            ranked_list[item[0]] = ranked_list[1]
        return self.evaluate(ranked_list)
        
def main():
    parser = argparse.ArgumentParser(description='Table Re-ranking')
    parser.add_argument('--model', type=str, required=True, choices=['openai','llama-3.3-70b-versatile', 'qwen-2.5-32b'], help='Base retrieval model')
    parser.add_argument('--dataset', type=str, required=True, choices=['spider', 'atis', 'geoq'], help='Dataset to process')
    parser.add_argument('--ranking_method', type=str, required=True, choices=['schema_only', 'random', 'relevant'], help='Re-ranking method to use')
    parser.add_argument('--output_dir', type=str, default='./reranker/outputs', help='Directory to save reranked ids')
    parser.add_argument('--eval_dir', type=str, default='./reranker/evaluation', help='Directory to save evaluation results')
    parser.add_argument('--num_candidates', type=int, default=10, choices=[10, 20, 30, 40, 50], help='Number of candidate tables to use')
    
    args = parser.parse_args()

    # If using the 'relevant' ranking method, load the huge FAISS index here once.
    if args.ranking_method == "relevant":
        embedder = OpenAIEmbeddings(model='text-embedding-3-small')
        row_db_path = os.path.join(args.output_dir, f"{args.dataset}_rowdb/")
        global row_db
        if os.path.exists(row_db_path):
            row_db = FAISS.load_local(row_db_path, embedder, allow_dangerous_deserialization=True)
        else:
            # Build row corpus and create the FAISS index.
            table_contents = load_pickle_file(f"./data/{args.dataset}/table_contents.pkl")
            table_dfs = convert_table_contents_to_dfs(table_contents)
            row_corpus = build_row_corpus(table_dfs)

            batch_size = 500
            max_workers = 4
            batches = [row_corpus[i:i + batch_size] for i in range(0, len(row_corpus), batch_size)]

            @retry(
                retry=retry_if_exception_type(RateLimitError),
                wait=wait_random_exponential(min=1, max=60),
                stop=stop_after_attempt(8),
            )
            def embed_batch(batch):
                texts = [doc.page_content for doc in batch]
                vectors = embedder.embed_documents(texts)
                return list(zip(texts, vectors)), [doc.metadata for doc in batch]

            text_embeddings, metadatas = [], []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(embed_batch, batch) for batch in batches]
                for future in tqdm(as_completed(futures), total=len(futures), desc='Embedding row corpus'):
                    batch_text_embeddings, batch_metadatas = future.result()
                    text_embeddings.extend(batch_text_embeddings)
                    metadatas.extend(batch_metadatas)

            row_db = FAISS.from_embeddings(text_embeddings, embedder, metadatas=metadatas)
            row_db.save_local(row_db_path)
        print("Global row_db loaded successfully.")

    # Initialize the appropriate re-ranker.
    rerankers = {
        'schema_only': lambda: SchemaReranker(
            args.model, args.dataset, args.ranking_method, args.output_dir, args.eval_dir, args.num_candidates
        ),
        'random': lambda: RandomRowsReranker(
            args.model, args.dataset, args.ranking_method, args.output_dir, args.eval_dir, args.num_candidates
        ),
        'relevant': lambda: RelevantRowsReranker(
            args.model, args.dataset, args.ranking_method, args.output_dir, args.eval_dir, args.num_candidates
        )
    }
    
    # Initialize the re-ranker.
    reranker = rerankers[args.ranking_method]()
    
    # Perform re-ranking.
    print(f"Re-ranking tables using {args.ranking_method} method...")
    reranker.rerank_and_eval()
    
if __name__ == "__main__":
    main()
