# DBRAG

DBRAG is a pipeline for retrieving, ranking, and question answering on that requires aggregating information from single or multiple tables.

## Setup and Installation

### 1. Create a Conda Environment and Install Requirements
Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys
Create a `.env` file in the project directory and add your API keys:

```plaintext
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

### 3. Data Processing
Generate the required data from scratch, follow these steps:

#### a. Process the Dataset
```bash
python3 process_dataset.py --dataset 'dataset_name'
```
Supported dataset options: `spider`, `atis`, `geoQuery`

This command generates a `data/` folder with subfolders:
- `spider/`
- `atis/`
- `geoq/`

#### b. Run the Retriever
```bash
python3 retriever.py --model 'model_name' --dataset 'dataset_name'
```
Supported models: `openai`, `contriever`, `tapas`

This command generates a `retriever/` folder with two subfolders:
- `outputs/`
- `evaluation/`

#### c. Run the Reranker
```bash
python3 reranker.py --model 'model_name' --dataset 'dataset_name' --ranking_method 'ranking_method'
```
Supported reranker models: `openai`, `llama-3.3-70b-versatile`, `qwen-2.5-32b`

Supported ranking methods: `schema_only`, `random`, `relevant`

This command generates a `reranker/` folder with two subfolders:
- `outputs/`
- `evaluation/`

### 4. Run the Reader
Finally, execute the DB-RAG module:
```bash
python3 dbrag.py --dataset 'dataset_name' --method 'method' --table_ids_path path_to_reranked_table_ids
```
Supported methods:
- `read_all`
- `read_k`
- `schema_only`
- `schema_random`
- `schema_relevant`

This command outputs a `reader/` folder with:
- `outputs/`
- `evaluation/`

## Running the JAR Baseline

The JAR scripts use paths relative to the `jar/` directory and expect the JAR-formatted
Spider files under `jar/data/{dataset}/`, including:

- `dev.json`
- `dev_tables.json`
- `dev_database/`
- `db_id_to_table_name_map.json`

The evaluator also requires the processed validation file at
`data/{dataset}/validation_data.pkl`.

The prediction file must be a JSON list with one list of table names per question. To
evaluate predictions at `k=5` for spider, run this command from the repository root:

```bash
python3 jar/jar_eval.py \
  --pred_path jar/data/spider/path/to/jar_predictions.json \
  --val_data_path data/spider/validation_data.pkl \
  --db_map_path jar/data/spider/db_id_to_table_name_map.json \
  --output_dir jar/evaluation \
  --k 5
```

The metrics are printed to the terminal and saved to:
`jar/evaluation/evaluation_results_jar_predictions.json`.

### Generating JAR Predictionss

1. Configure the dataset, model, and paths in `jar/contriever.py` (or
  `jar/openai_embed.py` / `jar/tapas.py`) and run the selected script to generate
	embeddings and scores.
2. Configure the corresponding settings and uncomment the prediction-generation call
  in `jar/ilp.py`. From the `jar/` directory, run one process per partition, for
  example:

```bash
cd jar
python3 ilp.py --partition 0
```

3. Run the merge step in `jar/ilp.py` after all partitions finish, then evaluate the
  merged prediction file with `jar/jar_eval.py` from the repository root. The
  evaluator's explicit paths in the command above avoid relying on its defaults.

The ILP stage requires the compatibility files referenced by `jar/compatibility.py`
(`dev_jaccard.json`, `dev_uniqueness.json`, `semantic_col_sim.json`, and
`exact_col_sim.json`) before prediction generation can run.

## Notes
- Ensure all dependencies are installed before running any script.
- The `.env` file is required for API authentication.
- Make sure to download or generate the required data before running the pipeline.
- The JAR scripts currently contain dataset- and model-specific settings in their
  `__main__` blocks; update those settings before running a different dataset or model.

## License
This project is licensed under the MIT License.

