from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig
from utils.pickle_utils import load_pickle_file
import numpy as np
import pandas as pd
from transformers.models.bart.modeling_bart import shift_tokens_right
from tqdm import tqdm

# Load the dataset
dataset = load_from_disk("./mtqa_data/atis/tokenized_atis_nq_test_with_answer.hf")
# Load BART tokenizer & TAPEX model config
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
config = AutoConfig.from_pretrained(tokenizer.name_or_path)

table_ids_list = np.load('./reranking_outputs/openai_atis_relevant_10c_reranked_ids.npy').tolist()
all_table_contents = load_pickle_file('./preprocessed_inputs/atis/table_contents.pkl')
all_table_names = load_pickle_file('./preprocessed_inputs/atis/table_names.pkl')
val_data = load_pickle_file('./preprocessed_inputs/atis/validation_data.pkl')


# Function to serialize tables (DataFrames) into structured text
def serialize_tables(table_names, tables):
    """
    Convert DataFrame tables into a structured text format.
    """
    serialized = []
    for table_name, table_df in zip(table_names, tables):
        header = " | ".join(table_df.columns)
        rows = [" | ".join(map(str, row)) for row in table_df.itertuples(index=False, name=None)]
        table_str = f"<table_name> : {table_name} col : {header} " + " ".join(f"row {i+1} : {r}" for i, r in enumerate(rows))
        serialized.append(table_str)
    return " ".join(serialized)

cnt = 0
cnt1 = 0
train_dataset = dataset["train"]
# Create a new list to store modified dataset
new_train_data = []
# Iterate through validation data and train dataset
for i, (val, train) in tqdm(enumerate(zip(val_data, train_dataset)), total=len(train_dataset)):
    assert val['question'] == train['question']  # Ensure questions match

    # Step 1: Get predicted table names & contents
    pred_table_ids = table_ids_list[i]
    tab_names, tab_contents , raw_tab_contents = [], [], []
    for tab_id in pred_table_ids:
        tab_names.append(all_table_names[tab_id])
        tab_contents.append(pd.DataFrame(all_table_contents[tab_id]['data'], columns=all_table_contents[tab_id]['columns']))
        raw_tab_contents.append(str(all_table_contents[tab_id]))

    # Step 2: Serialize tables
    serialized_tab = serialize_tables(tab_names, tab_contents)

    new_source= f"{train['question']} {serialized_tab}"

    if new_source != train["source"]:
        cnt += 1
    
    if not set(train['table_names']).issubset(set(tab_names[:3])):
        print(train['table_names'], tab_names[:2], i)
        cnt1 += 1
    assert len(tab_contents) == len(tab_names) == len(raw_tab_contents)


    # Step 4: Tokenize input using BART tokenizer (Lowercased + Remove quotes)
    input_encoding = tokenizer(
        new_source.strip().lower().replace('"', ''),
        return_tensors="pt",
        padding="max_length",
        max_length=1024,
        truncation="longest_first",
        add_special_tokens=True
    )

    # Step 5: Tokenize target (`labels`) using BART tokenizer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text=train["target"].strip().lower().replace('"', ''),
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=1024, 
            truncation="longest_first"
        )

    # Step 6: Compute shifted decoder input IDs for BART
    decoder_input_ids = shift_tokens_right(
        labels["input_ids"], tokenizer.pad_token_id, config.decoder_start_token_id
    )

    # Step 7: Append modified sample to new dataset list
    new_train_data.append({
        "query": train["query"],
        "question": train["question"],
        "answer": train["answer"],
        "table_names": tab_names,
        "tables": raw_tab_contents,
        "source": new_source,
        "target": train["target"],
        "input_ids": input_encoding["input_ids"].tolist(),
        "attention_mask": input_encoding["attention_mask"].tolist(),
        "labels": labels["input_ids"].tolist(),
        "decoder_input_ids": decoder_input_ids.tolist(),
    })

#Convert modified data back into a Hugging Face Dataset
new_train_dataset = Dataset.from_list(new_train_data)

# Save the new dataset with "train" key while keeping the old dataset intact
new_dataset = DatasetDict({"train": new_train_dataset})
new_dataset.save_to_disk("./mtqa_data/tokenized_spider_nq_test_with_answer_updated.hf")
print(f"{cnt} samples are different.")
print(f"{cnt1} samples have different table names.")

print("New dataset saved successfully at './mtqa_data/tokenized_spider_nq_test_with_answer_updated.hf'!")