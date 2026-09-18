import pandas as pd
from datasets import load_dataset, dataset_dict
import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set, Any
from utils.pickle_utils import load_pickle_file, save_as_pickle
from utils.table_utils import replace_with_table_name

class BaseTabQA(ABC):
    """Base class for Table Question Answering datasets."""
    
    def __init__(self, dataset_name: str, output_dir: str):
        self.dataset_name = dataset_name
        self.output_dir = f'{output_dir}/geoq' if dataset_name == "geoQuery" else f'{output_dir}/{dataset_name}'
        self.dataset = None
        self.stats = None
        os.makedirs(self.output_dir, exist_ok=True)
        

    @abstractmethod
    def load_dataset(self) -> None:
        """Load the dataset from the source."""
        pass

    def initialize_stats(self) -> Dict:
        """Initialize statistics tracking dictionary."""
        return {
            "table_counts": Counter(),
            "row_counts": [],
            "column_counts": [],
            "total_tables": 0,
            "total_questions": 0,
            "unique_table_names": set(),
            "unique_table_hashes": {},
            "unique_questions": set(),
        }

    def generate_table_hash(self, table_json: str) -> int:
        """Generate a hash of table content."""
        return hash(table_json)

    def save_stats(self) -> None:
        """Save statistics to CSV file."""
        if self.stats:
            json_path = os.path.join(self.output_dir, f"{self.dataset_name}_dataset_stats.json")
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(self.stats, json_file, indent=4)

    def save_processed_data(self, data: Any, filename: str) -> None:
        """Save processed data to pickle file."""
        save_as_pickle(data, os.path.join(self.output_dir, filename))

    def update_stats(self, stats_data: Dict, dataset: Dict) -> None:
        """Update statistics for dataset."""
        stats_data["total_questions"] += 1
        stats_data["unique_questions"].add(dataset['question'])
        stats_data["unique_table_names"].update(dataset['table_names'])

        num_tables = len(dataset['table_names'])
        stats_data["table_counts"][num_tables] += 1
        stats_data["total_tables"] += num_tables

        for table_json in dataset['tables']:
            table_hash = self.generate_table_hash(table_json)
            if table_hash not in stats_data["unique_table_hashes"]:
                table_data = json.loads(table_json)
                stats_data["unique_table_hashes"][table_hash] = {
                    "rows": len(table_data['index']),
                    "columns": len(table_data['columns'])
                }
            table_data = json.loads(table_json)
            stats_data["row_counts"].append(len(table_data['index']))
            stats_data["column_counts"].append(len(table_data['columns']))

    def prepare_validation_data(self, dataset: Dict) -> Dict:
        """Prepare validation data dictionary."""
        val_dict = {
            "question": dataset['question'].strip(),
        }
        temp_dic = json.loads(dataset['answer'])
        temp_dic['columns'] = [
            replace_with_table_name(cols.strip(), dataset['table_names']) 
            for cols in temp_dic['columns']
        ]
        val_dict["answer"] = temp_dic
        val_dict["table_ids"] = []
        return val_dict

    def process_tables(self, dataset: Dict, split: str, tables_hash_dict: Dict, 
                      table_contents: List, table_names: List, 
                      hash_to_index: Dict, id_count: int, diff: int, val_dict: Dict, test_name:str) -> Tuple[int, int]:
        """Process tables in the dataset."""
        for i, table_name in enumerate(dataset['table_names']):
            table_name = table_name.lower().strip()
            curr_table = json.loads(dataset['tables'][i])
            table_hash = self.generate_table_hash(dataset['tables'][i])

            if table_name not in tables_hash_dict or table_hash not in tables_hash_dict[table_name]:
                tables_hash_dict.setdefault(table_name, set()).add(table_hash)
                hash_to_index[table_hash] = id_count
                table_contents.append(curr_table)
                table_names.append(table_name)
                if split == test_name and val_dict:
                    diff += 1
                    val_dict["table_ids"].append(id_count)
                id_count += 1
            elif split == test_name and val_dict:
                val_dict["table_ids"].append(hash_to_index[table_hash])

        return id_count, diff

    def compile_stats(self, stats_data: Dict) -> Dict:
        """Compile final statistics from collected data."""
        row_counts = stats_data["row_counts"]
        column_counts = stats_data["column_counts"]

        return {
            "Total questions/dataset count": stats_data["total_questions"],
            "Total unique questions": len(stats_data["unique_questions"]),
            "Total unique tables (by table name)": len(stats_data["unique_table_names"]),
            "Total unique tables (by content)": len(stats_data["unique_table_hashes"]),
            "1-table questions": stats_data["table_counts"].get(1, 0),
            "2-table questions": stats_data["table_counts"].get(2, 0),
            "3-table questions": stats_data["table_counts"].get(3, 0),
            "4-table questions": stats_data["table_counts"].get(4, 0),
            "5-table questions": stats_data["table_counts"].get(5, 0),
            "Average number of rows": round(sum(row_counts) / len(row_counts), 2) if row_counts else 0,
            "Minimum number of rows": min(row_counts) if row_counts else 0,
            "Maximum number of rows": max(row_counts) if row_counts else 0,
            "Average number of columns": round(sum(column_counts) / len(column_counts), 2) if column_counts else 0,
            "Minimum number of columns": min(column_counts) if column_counts else 0,
            "Maximum number of columns": max(column_counts) if column_counts else 0,
        }

    @abstractmethod
    def process_dataset(self, generate_stats: bool = False) -> Tuple:
        """Process the dataset and return processed components."""
        pass

class SpiderTabQA(BaseTabQA):
    """Spider dataset implementation."""
    
    def load_dataset(self) -> None:
        """Load Spider dataset."""
        self.dataset = load_dataset(f'vaishali/{self.dataset_name}-tableQA')

    def process_dataset(self, generate_stats: bool = False) -> Tuple:
        """Process Spider dataset."""
        tables_hash_dict = defaultdict(set)
        sql_code = []
        table_contents = []
        table_names = []
        val_data = []
        hash_to_index = {}
        id_count = 0
        diff = 0
        test_name='validation'

        stats_data = self.initialize_stats() if generate_stats else None

        print(f"Processing {self.dataset_name} dataset...")
        for split in ['train', 'validation']:
            for dataset in tqdm(self.dataset[split], desc=f"Processing {split} data"):
                if split == test_name:
                    sql_code.append(dataset['query'])
                    val_dict = self.prepare_validation_data(dataset)
                else:
                    val_dict = None

                id_count, diff = self.process_tables(
                    dataset, split, tables_hash_dict, table_contents, table_names,
                    hash_to_index, id_count, diff, val_dict, test_name
                )
                
                if split == test_name and val_dict:
                    val_data.append(val_dict)

                if generate_stats:
                    self.update_stats(stats_data, dataset)

        self.stats = self.compile_stats(stats_data) if generate_stats else None
        
        return (table_contents, table_names, val_data, sql_code, self.stats)

class AtisTabQA(BaseTabQA):
    """ATIS dataset implementation."""
    
    def load_dataset(self) -> None:
        """Load ATIS dataset."""
        self.dataset = load_dataset(f'vaishali/{self.dataset_name}-tableQA')
    
    def process_dataset(self, generate_stats: bool = False) -> Tuple:
        """Process ATIS dataset - same format as Spider."""
        tables_hash_dict = defaultdict(set)
        sql_code = []
        table_contents = []
        table_names = []
        val_data = []
        hash_to_index = {}
        id_count = 0
        diff = 0
        test_name = 'test'

        stats_data = self.initialize_stats() if generate_stats else None

        print(f"Processing {self.dataset_name} dataset...")
        for split in ['train', 'validation', 'test']:
            for dataset in tqdm(self.dataset[split], desc=f"Processing {split} data"):
                if split == test_name:
                    sql_code.append(dataset['query'])
                    val_dict = self.prepare_validation_data(dataset)
                else:
                    val_dict = None

                id_count, diff = self.process_tables(
                    dataset, split, tables_hash_dict, table_contents, table_names,
                    hash_to_index, id_count, diff, val_dict, test_name
                )
                
                if split == test_name and val_dict:
                    val_data.append(val_dict)

                if generate_stats:
                    self.update_stats(stats_data, dataset)

        self.stats = self.compile_stats(stats_data) if generate_stats else None
        
        return (table_contents, table_names, 
                val_data, sql_code, self.stats)

class GeoTabQA(BaseTabQA):
    """GeoQuery dataset implementation."""
    
    def load_dataset(self) -> None:
        """Load GeoQuery dataset."""
        self.dataset = load_dataset(f'vaishali/{self.dataset_name}-tableQA')
    
    def process_dataset(self, generate_stats: bool = False) -> Tuple:
        """Process GeoQuery dataset - same format as Spider."""
        tables_hash_dict = defaultdict(set)
        sql_code = []
        table_contents = []
        table_names = []
        val_data = []
        hash_to_index = {}
        id_count = 0
        diff = 0
        test_name = 'test'

        stats_data = self.initialize_stats() if generate_stats else None

        print(f"Processing {self.dataset_name} dataset...")
        for split in ['train', 'validation', 'test']:
            for dataset in tqdm(self.dataset[split], desc=f"Processing {split} data"):
                if split == test_name:
                    sql_code.append(dataset['query'])
                    val_dict = self.prepare_validation_data(dataset)
                else:
                    val_dict = None

                id_count, diff = self.process_tables(
                    dataset, split, tables_hash_dict, table_contents, table_names,
                    hash_to_index, id_count, diff, val_dict, test_name
                )
                
                if split == test_name and val_dict:
                    val_data.append(val_dict)

                if generate_stats:
                    self.update_stats(stats_data, dataset)

        self.stats = self.compile_stats(stats_data) if generate_stats else None
        
        return (table_contents, table_names,
                val_data, sql_code, self.stats)

def main():
    parser = argparse.ArgumentParser(description='Process TableQA datasets')
    parser.add_argument('--dataset', type=str, choices=['spider', 'atis', 'geoQuery'], 
                      required=True, help='Dataset to process')
    parser.add_argument('--output_dir', type=str, default='./data',
                      help='Directory to save processed data')
    parser.add_argument('--generate_stats', action='store_true',
                      help='Generate dataset statistics')
    args = parser.parse_args()

    # Set up dataset processor based on selection
    dataset_processors = {
        'spider': SpiderTabQA,
        'atis': AtisTabQA,
        'geoQuery': GeoTabQA
    }

    processor_class = dataset_processors[args.dataset]
    processor = processor_class(args.dataset, args.output_dir)
    
    # Process the dataset
    processor.load_dataset()
    results = processor.process_dataset(generate_stats=args.generate_stats)
    
    # Save results
    if args.generate_stats:
        processor.save_stats()
    
    # Save processed data - same for all datasets since they share the same format
    table_contents, table_names, val_data, sql_code, _ = results
    assert len(table_contents) == len(table_names)
    assert len(val_data) == len(sql_code)
    processor.save_processed_data(sql_code, "sql_code.pkl")
    processor.save_processed_data(table_contents, "table_contents.pkl")
    processor.save_processed_data(table_names, "table_names.pkl")
    processor.save_processed_data(val_data, "validation_data.pkl")

if __name__ == "__main__":
    main()