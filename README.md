# DBRAG

DBRAG is a pipeline for retrieving, ranking, and question answering on that requires aggregating information from several tables.

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

## Notes
- Ensure all dependencies are installed before running any script.
- The `.env` file is required for API authentication.
- Make sure to download or generate the required data before running the pipeline.
- Evaluation results are written to the `evaluation/` subfolder for each pipeline stage.

## License
This project is licensed under the MIT License.

