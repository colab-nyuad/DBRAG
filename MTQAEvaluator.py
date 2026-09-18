import argparse
from pathlib import Path
from typing import Dict, Any
from QAEvaluator import MTQAEval
from utils.pickle_utils import load_pickle_file
import os

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MTQA Evaluation Script')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Name of the dataset to evaluate'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['full_retrieved', 'gold_tables'],
        default='full_retrieved',
        help='Evaluation method to use'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./mtqa_evaluation',
        help='Directory to save evaluation results'
    )
    
    return parser.parse_args()

def load_evaluation_data(base_path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load validation data and SQL code from pickle files.
    
    Args:
        base_path: Base path where pickle files are stored
        
    Returns:
        Tuple containing validation data and SQL code dictionaries
    """
    validation_data = load_pickle_file(base_path / 'validation_data.pkl')
    sql_code = load_pickle_file(base_path / 'sql_code.pkl')
    return validation_data, sql_code

def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the MTQA evaluation pipeline.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary containing evaluation results
    """
    # Setup paths
    base_path = Path('./data') / args.dataset
    output_file = Path('./mtqa_data') / f'{args.dataset}_with_{args.method}.txt'
    
    # Load data
    val_data, sql_code = load_evaluation_data(base_path)

    # create evaluation directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = MTQAEval(
        args.dataset,
        args.method,
        "mtqa",
        args.output_dir,
        with_column_names=False
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        val_data,
        sql_code,
        file_path=str(output_file)
    )
    
    # Save results
    evaluator.save_results(results)
    
    return results

def main():
    """Main entry point for the evaluation script."""
    args = parse_arguments()
    results = run_evaluation(args)
    print(f"Evaluation completed. Results saved to {args.output_dir}")
    return results

if __name__ == "__main__":
    main()