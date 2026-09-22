from typing import Dict, List, Union, Tuple
from collections import Counter, defaultdict
import re
from dataclasses import dataclass
from pathlib import Path
import json
from rouge_score import rouge_scorer, scoring
from datasets import load_from_disk
from evaluate import load as evaluate_load
from abc import ABC, abstractmethod
from tqdm import tqdm
import tiktoken

@dataclass
class GroupMetrics:
    exact_match: float
    count: int
    micro_metrics: Dict[str, Dict[str, float]]
    macro_metrics: Dict[str, Dict[str, float]]

@dataclass
class EvaluationResults:
    exact_match: float
    micro_metrics: Dict[str, Dict[str, float]]
    macro_metrics: Dict[str, Dict[str, float]]
    grouped_by_tables: Dict[int, GroupMetrics]
    grouped_by_predicate: Dict[int, GroupMetrics]

class BaseTableEvaluator(ABC):
    def __init__(self, dataset_name: str, reader_method: str, method: str, output_dir: str, truncate: bool = True, with_column_names: bool = False):
        self.dataset_name = dataset_name
        self.method = method
        self.reader_method = reader_method
        self.output_dir = Path(output_dir)
        self.truncate = truncate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.with_column_names = with_column_names
        
        # Initialize metrics
        self.rouge_types = ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(rouge_types=self.rouge_types, use_stemmer=True)
        self.exact_match_metric = evaluate_load("exact_match")

    def extract_predictions(self, file_path):
        predictions = []
        with open(file_path, 'r') as file:
            for line in file:
                if "prediction:" in line:
                    # Extract the prediction value using regex
                    match = re.search(r'prediction:\s*(.*)', line)
                    if match:
                        if self.with_column_names:
                            predictions.append(match.group(1).strip())
                        else:
                            curr_text = match.group(1).strip()
                            text_match = re.search(r"row\s*\d+\s*:.*", curr_text)
                            if text_match:
                                predictions.append(text_match.group(0))
                            else:
                                print(f"Could not extract column names from: {curr_text}")
        return predictions

    def truncate_text(self, text: str, max_toks: int = 1024) -> List[str]:
        """
        Truncate a text to fit within the token limit of the embedding model.
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        if len(tokens) > max_toks:
            print(f"Truncating text from {len(tokens)} tokens to {max_toks} tokens.")
            tokens = tokens[:max_toks]
        return tokenizer.decode(tokens)
        
    def serialize_input(self, columns: List[str], rows: List[List[str]]) -> str:
        """Serialize columns and rows into a standard format."""
        columns = [str(col) for col in columns]
        rows = [[str(val) for val in row] for row in rows]
        if self.with_column_names:
            final_str = "col : " + " | ".join(columns) + " "
        else:
            final_str = ""
        
        for ind, row in enumerate(rows):
            final_str += f"row {ind+1} : " + " | ".join(row) + " "

        if self.truncate:
            final_str = self.truncate_text(final_str)
            
        return final_str.strip().lower() if len(columns) > 0 else ""
    
    def get_rows_columns_cells(self, line: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract rows, columns and cells from serialized string."""
        line = line.lower()
        if self.with_column_names:
            line = line.split("col :")[1].strip() 
            lines = re.split(r"\s+row\s+[0-9]+\s+:\s+", line)
            rows = [" | ".join([cell.strip() for cell in row.split("|")]) for row in lines[1:]]
            cells = [cell.strip() for row in lines[1:] for cell in row.split("|")]
            columns = [" | ".join([elem.strip() for elem in elems]) for elems in list(zip(*[row.split(" | ") for row in lines]))]
        else:
            lines = re.split(r"\s*row\s+\d+\s*:\s*", line)[1:]  # Split based on row pattern
            rows = [row.strip() for row in lines]
            cells = [cell.strip() for row in rows for cell in row.split(" | ")]  
            # Transpose the row-wise data to get columns
            columns = [" | ".join(col).strip() for col in zip(*[row.split(" | ") for row in rows])]
        return rows, columns, cells
    
    def get_correct_total_prediction(self, target_str: str, pred_str: str) -> Dict:
        """Compare target and prediction strings to get statistics."""
        if pred_str == '':
            pred_str = "col :"
            
        target_rows, target_columns, target_cells = self.get_rows_columns_cells(target_str)
        prediction_rows, prediction_columns, prediction_cells = self.get_rows_columns_cells(pred_str)
        
        common_rows = Counter(target_rows) & Counter(prediction_rows)
        common_columns = Counter(target_columns) & Counter(prediction_columns)
        common_cells = Counter(target_cells) & Counter(prediction_cells)
        
        return {
            "target_rows": target_rows,
            "target_columns": target_columns,
            "target_cells": target_cells,
            "pred_rows": prediction_rows,
            "pred_columns": prediction_columns,
            "pred_cells": prediction_cells,
            "correct_rows": list(common_rows.elements()),
            "correct_columns": list(common_columns.elements()),
            "correct_cells": list(common_cells.elements())
        }

    def calculate_group_metrics(self, ground_truth: List[str], predictions: List[str]) -> Tuple[Dict, Dict]:
        """Calculate micro and macro metrics for a group."""
        total_stats = defaultdict(int)
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        
        for gt, pred in zip(ground_truth, predictions):
            stats = self.get_correct_total_prediction(gt.strip(), pred.strip())
            
            # Update totals for micro metrics
            for key in ['rows', 'columns', 'cells']:
                total_stats[f'total_{key}'] += len(stats[f'target_{key}'])
                total_stats[f'total_predicted_{key}'] += len(stats[f'pred_{key}'])
                total_stats[f'total_correct_{key}'] += len(stats[f'correct_{key}'])
            
            # Calculate individual precision and recall for macro metrics
            for key in ['rows', 'columns', 'cells']:
                pred_len = len(stats[f'pred_{key}'])
                target_len = len(stats[f'target_{key}'])
                correct_len = len(stats[f'correct_{key}'])
                
                precisions[key].append(correct_len / pred_len if pred_len > 0 else 0)
                recalls[key].append(correct_len / target_len if target_len > 0 else 0)
        
        # Calculate micro metrics
        micro_metrics = {}
        for key in ['rows', 'columns', 'cells']:
            precision = total_stats[f'total_correct_{key}'] / total_stats[f'total_predicted_{key}'] if total_stats[f'total_predicted_{key}'] > 0 else 0
            recall = total_stats[f'total_correct_{key}'] / total_stats[f'total_{key}'] if total_stats[f'total_{key}'] > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
            micro_metrics[key] = {'precision': round(precision*100, 2), 'recall': round(recall*100, 2), 'f1': round(f1*100, 2)}
        
        # Calculate macro metrics
        macro_metrics = {}
        for key in ['rows', 'columns', 'cells']:
            precision = sum(precisions[key]) / len(precisions[key]) if precisions[key] else 0
            recall = sum(recalls[key]) / len(recalls[key]) if recalls[key] else 0
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
            macro_metrics[key] = {'precision': round(precision*100, 2), 'recall': round(recall*100, 2), 'f1': round(f1*100, 2)}
        
        return micro_metrics, macro_metrics
    
    def _calculate_grouped_metrics(self, grouped_data: Dict[int, List[Tuple[str, str]]], grouped_type_str:str) -> Dict[int, GroupMetrics]:
        """Calculate detailed metrics for grouped data."""
        results = {}
        for group_id, pairs in tqdm(grouped_data.items(), desc=f"Computing grouped metrics({grouped_type_str})"):
            ground_truth, predictions = zip(*pairs)
            
            # Calculate exact match
            em_scores = [
                self.exact_match_metric.compute(predictions=[pred.strip()], references=[gt.strip()])['exact_match']
                for gt, pred in zip(ground_truth, predictions)
            ]
            
            # Calculate micro and macro metrics for the group
            micro_metrics, macro_metrics = self.calculate_group_metrics(ground_truth, predictions)
            
            results[group_id] = GroupMetrics(
                exact_match=round(sum(em_scores) / len(em_scores) * 100, 2),
                count=len(pairs),
                micro_metrics=micro_metrics,
                macro_metrics=macro_metrics
            )
        return results
    
    def calculate_metrics(self, ground_truth: List[str], predictions: List[str], 
                         table_counts: List[int], has_predicates: List[bool]) -> EvaluationResults:
        """Calculate all evaluation metrics."""
        # Calculate overall metrics
        micro_metrics, macro_metrics = self.calculate_group_metrics(ground_truth, predictions)
        
        # Calculate exact match
        em_scores = [
            self.exact_match_metric.compute(predictions=[pred.strip()], references=[gt.strip()])['exact_match']
            for gt, pred in tqdm(zip(ground_truth, predictions), total=len(predictions), desc="Computing overall metrics")
        ]
        
        # Group by tables and predicates
        grouped_by_tables = defaultdict(list)
        grouped_by_predicate = defaultdict(list)
        
        for gt, pred, table_count, has_pred in zip(ground_truth, predictions, table_counts, has_predicates):
            grouped_by_tables[table_count].append((gt, pred))
            grouped_by_predicate[1 if has_pred else 0].append((gt, pred))
        
        return EvaluationResults(
            exact_match=round(sum(em_scores) / len(em_scores) * 100, 2),
            micro_metrics=micro_metrics,
            macro_metrics=macro_metrics,
            grouped_by_tables=self._calculate_grouped_metrics(grouped_by_tables, "by tables"),
            grouped_by_predicate=self._calculate_grouped_metrics(grouped_by_predicate, "by predicates")
        )
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> EvaluationResults:
        """Evaluate the predictions and return metrics."""
        pass
    
    def save_results(self, results: EvaluationResults):
        """Save evaluation results to files."""
        output_path = self.output_dir / f"{self.dataset_name}_{self.method}_{self.reader_method}.json"
        
        def group_metrics_to_dict(group_metrics: Dict[int, GroupMetrics]) -> Dict:
            return {
                str(k): {
                    'exact_match': v.exact_match,
                    'count': v.count,
                    'micro_metrics': v.micro_metrics,
                    'macro_metrics': v.macro_metrics
                } for k, v in group_metrics.items()
            }
        
        print(f"Saving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump({
                'exact_match': results.exact_match,
                'micro_metrics': results.micro_metrics,
                'macro_metrics': results.macro_metrics,
                'grouped_by_tables': group_metrics_to_dict(results.grouped_by_tables),
                'grouped_by_predicate': group_metrics_to_dict(results.grouped_by_predicate)
            }, f, indent=2)

class MultiTableRAGEval(BaseTableEvaluator):
    def evaluate(self, predictions: Dict[int, Dict], val_data: List[Dict], sql_code: List[str]) -> EvaluationResults:
        """Evaluate predictions from RAG system."""
        ground_truth = []
        predicted = []
        table_counts = []
        has_predicates = []
        
        print("Evaluation in progress:")

        for idx, pred_item in tqdm(sorted(predictions.items()), desc="Processing outputs"):
            # Process ground truth
            gold_item = pred_item['gold_answer']
            ground_truth.append(self.serialize_input(gold_item['columns'], gold_item['data']))
            
            # Process prediction
            pred_data = pred_item['final_answer']
            predicted.append(self.serialize_input(pred_data['columns'], pred_data['data']))
            
            # Get metadata
            table_counts.append(len(val_data[idx]['table_ids']))
            has_predicates.append(bool(re.search(r'\bWHERE\b', sql_code[idx], re.IGNORECASE)))
        
        # with open("ground_truth.csv", "w") as f:
        #     f.write("Question,Ground Truth,Prediction\n")
        #     for idx, (item, gt, pred) in enumerate(zip(val_data, ground_truth, predicted)):
        #         f.write(f"{item['question']},{gt},{pred}\n")

        return self.calculate_metrics(ground_truth, predicted, table_counts, has_predicates)

class MTQAEval(BaseTableEvaluator):
    def evaluate(self, val_data: List[Dict], sql_code: List[str], file_path: str) -> EvaluationResults:
        """Evaluate predictions from MTQA system."""
        ground_truth = []
        table_counts = []
        has_predicates = []
        predictions = self.extract_predictions(file_path)
        
        for idx, item in enumerate(val_data):
            ground_truth.append(self.serialize_input(item['answer']['columns'], item['answer']['data']))
            table_counts.append(len(item['table_ids']))
            has_predicates.append(bool(re.search(r'\bWHERE\b', sql_code[idx], re.IGNORECASE)))

        return self.calculate_metrics(ground_truth, predictions, table_counts, has_predicates)