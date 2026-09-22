import argparse
import torch
import numpy as np
import pandas as pd
import os, tiktoken
from abc import ABC, abstractmethod
from typing import List, Dict
from tqdm import tqdm
import warnings
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from openai import OpenAI
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import json
from utils.pickle_utils import load_pickle_file

load_dotenv()

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

class BaseTableRetriever(ABC):
    """Base class for table retrieval models."""
    
    def __init__(self, model_name: str, dataset_name: str, k: int, 
                 output_dir: str, eval_dir: str):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.k = k
        self.output_dir = output_dir
        self.eval_dir = eval_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Load dataset
        self.load_dataset()
    
        self.retrieved_ids_path = os.path.join(self.output_dir, f"{self.model_name}_{self.dataset_name}_retrieved_ids.npy")

    def load_dataset(self) -> None:
        """Load the processed dataset files."""
        dataset_dir = f"./data/{self.dataset_name}"
        self.table_contents = load_pickle_file(f"{dataset_dir}/table_contents.pkl")
        self.table_names = load_pickle_file(f"{dataset_dir}/table_names.pkl")
        self.validation_data = load_pickle_file(f"{dataset_dir}/validation_data.pkl")

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode input texts into embeddings."""
        pass

    @abstractmethod
    def embed_tables(self) -> np.ndarray:
        """Embed all tables in the dataset."""
        pass

    def embed_queries(self) -> np.ndarray:
        """Embed all validation queries."""
        queries = [data['question'] for data in self.validation_data]
        return self.encode(queries)

    def compute_and_save_embeddings(self) -> None:
        """Compute and save both table and query embeddings."""
        # Compute and save table embeddings if not already present
        if not os.path.exists(self.table_embeddings_path):
            print("Computing table embeddings...")
            table_embeddings = self.embed_tables()
            np.save(self.table_embeddings_path, table_embeddings)
        self.table_embeddings = np.load(self.table_embeddings_path)
        # Compute and save query embeddings if not already present
        if not os.path.exists(self.query_embeddings_path):
            print("Computing query embeddings...")
            query_embeddings = self.embed_queries()
            np.save(self.query_embeddings_path, query_embeddings)
        self.query_embeddings = np.load(self.query_embeddings_path)

    @abstractmethod
    def retrieve_all(self, max_k: int = 50) -> np.ndarray:
        """Retrieve top-k tables for all queries."""
        raise NotImplementedError("Subclass must implement this method")

    def compute_and_save_retrievals(self, max_k: int = 50) -> None:
        """Compute and save retrieved table ids for all queries."""
        if not os.path.exists(self.retrieved_ids_path):
            retrieved_ids = self.retrieve_all(max_k)
            np.save(self.retrieved_ids_path, retrieved_ids)
        self.retrieved_ids = np.load(self.retrieved_ids_path)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate retrieval performance on validation set, with dynamic splitting by 
        number of required tables and overall metrics.
        
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        queries_by_table_count = defaultdict(list)
        for i, query_data in enumerate(self.validation_data):
            num_tables = len(query_data['table_ids'])
            if num_tables > 0:
                queries_by_table_count[num_tables].append((i, query_data))
        
        table_counts = sorted(queries_by_table_count.keys())
        k_values = [5, 10, 20, 50]
        results = {"Total_Questions": {}}
        
        for num_tables in table_counts:
            results["Total_Questions"][f"{num_tables}-table(s)"] = len(queries_by_table_count[num_tables])
        
        metric_categories = ["Precision", "Recall", "F1"]
        grouped_metrics = {metric: defaultdict(dict) for metric in metric_categories}
        
        for num_tables in table_counts:
            queries = queries_by_table_count[num_tables]
            group_metrics = {f"recall@{k}": [] for k in k_values}
            group_metrics.update({f"precision@{k}": [] for k in k_values})
            group_metrics.update({f"f1@{k}": [] for k in k_values})
            
            for query_idx, query_data in queries:
                gold_table_ids = set(query_data['table_ids'])
                
                for k in k_values:
                    pred_table_ids = set(self.retrieved_ids[query_idx][:k])
                    n_correct = len(gold_table_ids.intersection(pred_table_ids))
                    
                    recall = n_correct / len(gold_table_ids) if gold_table_ids else 0
                    group_metrics[f"recall@{k}"].append(recall)
                    
                    precision = n_correct / k if k > 0 else 0
                    group_metrics[f"precision@{k}"].append(precision)
                    
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    group_metrics[f"f1@{k}"].append(f1)
            
            for k in k_values:
                grouped_metrics["Precision"][f"{num_tables}-table(s)"][f"@{k}"] = np.mean(group_metrics[f"precision@{k}"])
                grouped_metrics["Recall"][f"{num_tables}-table(s)"][f"@{k}"] = np.mean(group_metrics[f"recall@{k}"])
                grouped_metrics["F1"][f"{num_tables}-table(s)"][f"@{k}"] = np.mean(group_metrics[f"f1@{k}"])
        
        overall_metrics = {metric: {} for metric in metric_categories}
        all_queries_metrics = {f"recall@{k}": [] for k in k_values}
        all_queries_metrics.update({f"precision@{k}": [] for k in k_values})
        all_queries_metrics.update({f"f1@{k}": [] for k in k_values})
        
        for i, query_data in enumerate(self.validation_data):
            gold_table_ids = set(query_data['table_ids'])
            if not gold_table_ids:
                continue
            
            for k in k_values:
                pred_table_ids = set(self.retrieved_ids[i][:k])
                n_correct = len(gold_table_ids.intersection(pred_table_ids))
                
                recall = n_correct / len(gold_table_ids)
                all_queries_metrics[f"recall@{k}"].append(recall)
                
                precision = n_correct / k if k > 0 else 0
                all_queries_metrics[f"precision@{k}"].append(precision)
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                all_queries_metrics[f"f1@{k}"].append(f1)
        
        for k in k_values:
            overall_metrics["Precision"][f"@{k}"] = np.mean(all_queries_metrics[f"precision@{k}"])
            overall_metrics["Recall"][f"@{k}"] = np.mean(all_queries_metrics[f"recall@{k}"])
            overall_metrics["F1"][f"@{k}"] = np.mean(all_queries_metrics[f"f1@{k}"])
        
        results.update(grouped_metrics)
        results.update({"Overall": overall_metrics})
        
        results_path = os.path.join(self.eval_dir, f"{self.model_name}_{self.dataset_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2))
        
        return results

class OpenAIEmbeddingRetriever(BaseTableRetriever):
    """OpenAI text embedding based retriever."""
    
    def __init__(self, model_name: str, dataset_name: str, k: int, 
                 output_dir: str, eval_dir: str):
        super().__init__(model_name, dataset_name, k, output_dir, eval_dir)
        self.embed_batch_size = 2000  # OpenAI API rate limit consideration
        self.client = OpenAI()
        self.embedding_model = "text-embedding-3-small"
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 512
        self.table_embeddings_path = os.path.join(self.output_dir, f"{self.model_name}_{self.dataset_name}_table_embeddings.npy")
        self.query_embeddings_path = os.path.join(self.output_dir, f"{self.model_name}_{self.dataset_name}_query_embeddings.npy")
        
        # Load or compute embeddings
        self.compute_and_save_embeddings()
        # Load or compute retrievals
        self.compute_and_save_retrievals()

    def retrieve_all(self, max_k: int = 50) -> np.ndarray:
        """
        Retrieve top-k tables for all queries at once.
        
        Args:
            max_k (int): Maximum number of tables to retrieve per query
            
        Returns:
            np.ndarray: Array of shape (n_queries, max_k) containing retrieved table ids
        """
        # Calculate similarities for all queries at once
        similarities = cosine_similarity(self.query_embeddings, self.table_embeddings)

        # Get top-k indices for each query
        top_k_indices = np.argsort(similarities, axis=1)[:, -max_k:][:, ::-1]
    
        return top_k_indices

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI embeddings API."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.embed_batch_size),
                     desc="Encoding texts"):
            batch_texts = texts[i:i + self.embed_batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Error in OpenAI API call: {e}")
                # Return zero embeddings for failed batch
                embeddings = np.zeros((len(batch_texts), 1536))  # text-embedding-small-3 dimension
                all_embeddings.extend(embeddings)
                
        return np.array(all_embeddings)

    def embed_tables(self) -> np.ndarray:
        """Embed all tables using OpenAI embeddings."""
        # Prepare table texts
        table_texts = []
        for table_name, table_content in zip(self.table_names, self.table_contents):
            columns = [str(col) for col in table_content['columns']]
            sample_rows = [f"row {i+1}: " + " | ".join(map(str, row)) for i, row in enumerate(table_content['data'][:2])]
            table_text = f"table name: {table_name} col: {' | '.join(columns)} "
            table_text += " ".join(sample_rows)
            table_text = table_text.lower()
            # truncate if it exceeds max token using tiktoken
            tokens = self.tokenizer.encode(table_text)
            if len(tokens) > self.max_tokens:
                table_text = self.tokenizer.decode(tokens[:self.max_tokens])
            table_texts.append(table_text)
        
        return self.encode(table_texts)

class ContrieverRetriever(BaseTableRetriever):
    """Contriever (MS-MARCO) based retriever."""
    
    def __init__(self, model_name: str, dataset_name: str, k: int, 
                 output_dir: str, eval_dir: str):
        super().__init__(model_name, dataset_name, k, output_dir, eval_dir)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
        self.model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(self.device)
        self.max_length = 512
        self.table_embeddings_path = os.path.join(self.output_dir, f"{self.model_name}_{self.dataset_name}_table_embeddings.npy")
        self.query_embeddings_path = os.path.join(self.output_dir, f"{self.model_name}_{self.dataset_name}_query_embeddings.npy")
        
        # Load or compute embeddings
        self.compute_and_save_embeddings()
        # Load or compute retrievals
        self.compute_and_save_retrievals()

    def retrieve_all(self, max_k: int = 50) -> np.ndarray:
        """
        Retrieve top-k tables for all queries at once.
        
        Args:
            max_k (int): Maximum number of tables to retrieve per query
            
        Returns:
            np.ndarray: Array of shape (n_queries, max_k) containing retrieved table ids
        """
        # Calculate similarities for all queries at once
        similarities = cosine_similarity(self.query_embeddings, self.table_embeddings)

        # Get top-k indices for each query
        top_k_indices = np.argsort(similarities, axis=1)[:, -max_k:][:, ::-1]
    
        return top_k_indices

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Contriever model."""
        all_embeddings = []
        batch_size = 200  # Adjust based on GPU memory
        
        for i in tqdm(range(0, len(texts), batch_size),
                     desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                embeddings = self.mean_pooling(token_embeddings, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def mean_pooling(self, token_embeddings, mask):
        """Mean pooling with attention mask to ignore padding tokens."""
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return embeddings

    def embed_tables(self) -> np.ndarray:
        """Embed all tables using Contriever."""
        # Prepare table texts
        table_texts = []
        for table_name, table_content in zip(self.table_names, self.table_contents):
            columns = [str(col) for col in table_content['columns']]
            sample_rows = [f"row {i+1}: " + " | ".join(map(str, row)) for i, row in enumerate(table_content['data'][:2])]
            table_text = f"table name: {table_name} col: {' | '.join(columns)} "
            table_text += " ".join(sample_rows)
            table_text = table_text.lower()
            table_texts.append(table_text)
        
        return self.encode(table_texts)

class TAPASScore(nn.Module):

  def __init__(self, lm, device, dropout=0.2):
    super(TAPASScore, self).__init__()
    print(f'model lm: {lm}')
    self.tapas = AutoModel.from_pretrained(f'google/{lm}')
    self.dropout = nn.Dropout(dropout)
    self.device = device
    hidden_size = self.tapas.config.hidden_size
    self.linear1 = nn.Linear(hidden_size, 256)
    self.linear2 = nn.Linear(256, 1)
  
  def forward(self, x):
    output = self.dropout(self.tapas(**x).pooler_output)
    output = self.dropout(self.linear1(output))
    output = self.linear2(output)
    return output
  

class TapasDTRRetriever(BaseTableRetriever):
    """TAPAS DTR (Dense Table Retriever) based retriever."""
    
    def __init__(self, model_name: str, dataset_name: str, k: int, 
                 output_dir: str, eval_dir: str, checkpoint_path: str = None,
                 lm: str = 'tapas-large'):
        super().__init__(model_name, dataset_name, k, output_dir, eval_dir)
        
        # Initialize TAPAS scoring model
        self.scoring_model = TAPASScore(lm=lm, device=self.device)
        self.scoring_model.load_state_dict(torch.load(checkpoint_path))
        self.scoring_model.to(self.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f'google/{lm}')
        
        # Compute and save retrievals initially
        self.compute_and_save_retrievals()

    def prepare_batch(self, questions: List[str], table_data: Dict) -> torch.Tensor:
        """
        Prepare a batch of inputs for the model.
        
        Args:
            questions (List[str]): List of questions
            table_data (Dict): Table data to process
            
        Returns:
            torch.Tensor: Tokenized inputs
        """
        # Convert table to DataFrame format
        df = pd.DataFrame(
            data=table_data['data'][:2],  # Using first 2 rows as in original
            columns=table_data['columns']
        ).astype(str)
        
        # Tokenize inputs
        return self.tokenizer(
            table=df,
            queries=questions,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        pass

    def embed_tables(self) -> np.ndarray:
        pass

    def compute_scores(self) -> np.ndarray:
        """
        Compute scores for all query-table pairs using the DTR model.
        
        Returns:
            np.ndarray: Array of shape (n_queries, n_tables) containing scores
        """
        # Check if scores already exist
        scores_path = os.path.join(self.output_dir, f"{self.model_name}_{self.dataset_name}_dtr_scores.npy")
        if os.path.exists(scores_path):
            print("Loading pre-computed DTR scores...")
            return np.load(scores_path)
        
        print("Computing DTR scores...")
        self.scoring_model.eval()
        scores = []
        
        # Get all questions
        questions = [data['question'].strip() for data in self.validation_data]
        
        # Process each table
        with torch.no_grad():
            for table_content in tqdm(self.table_contents, desc="Computing scores"):
                # Prepare input batch
                batch_input = self.prepare_batch(questions, table_content)
                
                # Get scores
                output = self.scoring_model(batch_input)
                scores.append(output)
        
        # Combine all scores
        scores = torch.hstack(scores).cpu().numpy()
        
        # Save scores
        np.save(scores_path, scores)
        return scores

    def retrieve_all(self, max_k: int = 50) -> np.ndarray:
        """
        Retrieve top-k tables for all queries using DTR scores.
        
        Args:
            max_k (int): Maximum number of tables to retrieve per query
            
        Returns:
            np.ndarray: Array of shape (n_queries, max_k) containing retrieved table ids
        """
        # Compute or load scores
        scores = self.compute_scores()
        
        # Get top-k indices for each query
        top_k_indices = np.argsort(scores, axis=1)[:, -max_k:][:, ::-1]
        
        return top_k_indices


def main():
    parser = argparse.ArgumentParser(description='Table Retrieval Models')
    parser.add_argument('--model', type=str, 
                       choices=['openai', 'contriever', 'dtr'],
                       required=True, help='Retrieval model to use')
    parser.add_argument('--dataset', type=str,
                       choices=['spider', 'atis', 'geoq'],
                       required=True, help='Dataset to process')
    parser.add_argument('--k', type=int, default=50,
                       help='Maximum number of tables to retrieve')
    parser.add_argument('--output_dir', type=str, default='./retriever/outputs',
                       help='Directory to save embeddings')
    parser.add_argument('--eval_dir', type=str, default='./retriever/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--checkpoint_path', type=str, default='./tapas/checkpoint.pt',
                       help='Path to TAPAS DTR checkpoint')
    parser.add_argument('--lm', type=str, default='tapas-large',
                       help='TAPAS language model type')
    
    args = parser.parse_args()
    
    # Initialize appropriate retriever
    retrievers = {
        'openai': lambda: OpenAIEmbeddingRetriever(
            'openai', args.dataset, args.k, args.output_dir,  args.eval_dir
        ),
        'contriever': lambda: ContrieverRetriever(
            'contriever', args.dataset, args.k, args.output_dir, args.eval_dir
        ),
        'dtr': lambda: TapasDTRRetriever(
            'tapas-dtr', args.dataset, args.k, args.output_dir, args.eval_dir,
            checkpoint_path=args.checkpoint_path,
            lm=args.lm,
        )
    }
    
    retriever = retrievers[args.model]()
    
    # Evaluate and print results
    results = retriever.evaluate()

if __name__ == "__main__":
    main()