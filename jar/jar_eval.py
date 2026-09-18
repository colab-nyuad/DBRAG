import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set
import argparse
import pickle
import os

class JarEvaluator:
    def __init__(self, val_data_path: str, db_id_map_path: str):
        """
        Initialize the evaluator with validation data and database mapping.
        
        Args:
            val_data_path: Path to the validation data pickle file
            db_id_map_path: Path to the database ID to table name mapping JSON file
        """
        self.validation_data = self._load_pickle_file(val_data_path)
        self.db_id_map = self._load_json_file(db_id_map_path)
        
    def _load_pickle_file(self, path: str):
        """Load pickle file using the provided utility function."""
        return pickle.load(open(path, 'rb'))
    
    def _load_json_file(self, path: str) -> Dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
            
    def _load_predictions(self, pred_path: str) -> List[List[str]]:
        """
        Load and process predictions from JSON file.
        
        Args:
            pred_path: Path to the predictions file
            
        Returns:
            List of lists containing table names for each prediction
        """
        preds = self._load_json_file(pred_path)
            
        pred_ids = []
        for pred_list in preds:
            p = []
            for table in pred_list:
                p.append(self.db_id_map[table])
            pred_ids.append(p)
            
        return pred_ids
        
    def evaluate(self, pred_path: str, k: int = 5) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            pred_path: Path to the predictions file
            k: Number of top predictions to consider (default: 5)
            
        Returns:
            Dictionary containing metrics for each table count category and overall
        """
        pred_ids = self._load_predictions(pred_path)
        
        # Group queries by number of tables in ground truth
        queries_by_table_count = defaultdict(list)
        for i, query_data in enumerate(self.validation_data):
            num_tables = len(query_data['table_ids'])
            if num_tables > 0:
                queries_by_table_count[num_tables].append(i)
                
        metrics_by_table_count = defaultdict(lambda: defaultdict(list))
        overall_metrics = defaultdict(list)
        
        # Calculate metrics for each group
        for num_tables, query_indices in sorted(queries_by_table_count.items(), key=lambda x: x[0]):
            for i in query_indices:
                query_data = self.validation_data[i]
                gold_table_ids = set(query_data['table_ids'])
                
                if not gold_table_ids:
                    continue
                    
                pred_table_ids = set(pred_ids[i][:k])
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
        
        # Format results
        results = {}
        for num_tables, metrics in metrics_by_table_count.items():
            results[num_tables] = {
                'precision@5': float(np.mean(metrics['precision'])),
                'recall@5': float(np.mean(metrics['recall'])),
                'f1@5': float(np.mean(metrics['f1']))
            }
            
        results['overall'] = {
            'precision@5': float(np.mean(overall_metrics['precision'])),
            'recall@5': float(np.mean(overall_metrics['recall'])),
            'f1@5': float(np.mean(overall_metrics['f1']))
        }
        
        return results

    def save_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary containing evaluation results
            output_path: Path where to save the results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate table prediction results')
    parser.add_argument('--pred_path', type=str, required=True,
                      help='Path to the predictions JSON file')
    parser.add_argument('--val_data_path', type=str, 
                      default='../data/spider/validation_data.pkl',
                      help='Path to validation data pickle file')
    parser.add_argument('--db_map_path', type=str,
                      default='./data/spider/db_id_to_table_name_map.json',
                      help='Path to database ID to table name mapping JSON file')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                      help='Directory to save evaluation results')
    parser.add_argument('--k', type=int, default=5,
                      help='Number of top predictions to consider (default: 5)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = JarEvaluator(
        val_data_path=args.val_data_path,
        db_id_map_path=args.db_map_path
    )
    
    # Run evaluation
    results = evaluator.evaluate(args.pred_path, k=args.k)
    
    # Generate output path
    output_path = os.path.join(
        args.output_dir,
        f"evaluation_results_{os.path.basename(args.pred_path)}"
    )
    
    # Save results
    evaluator.save_results(results, output_path)
    
    # Print results
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()