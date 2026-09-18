import re, ast, json
from langchain.docstore.document import Document
from collections import Counter
import pandas as pd
from tqdm import tqdm
import numpy as np

def table_to_text(columns, rows, table_name):
    """
    Convert a table into a serialized text format for embedding.
    
    Args:
        columns (list of str): The column names of the table.
        rows (list of list): The rows of the table.
        table_name (str): The name of the table.
    
    Returns:
        str: Serialized text representation of the table.
    """
    # Convert columns to lowercase strings
    columns = [str(column).lower() for column in columns]

    # Convert each cell to a UTF-8 encoded string, handling special characters
    rows = [[str(cell).lower() for cell in row] for row in rows]

    # Create header and rows text
    header = " | ".join(columns)
    rows_text = [f"row {i+1}: " + " | ".join(row) for i, row in enumerate(rows)]

    # Combine into a single serialized string
    serialized_table = f"table name: {table_name} cols: {header} " + " ".join(rows_text)
    return serialized_table

# function to create a string representation of a table(for reranking)
def create_table_string_representation(table_id, table_name, columns,rows):
    table_string = f"table_id: {table_id}\ntable_name: {table_name}\n" if table_id else f"table_name: {table_name}\n"

    header = " | ".join([str(column) for column in columns])
    table_string += f"columns: {header}\n"

    for i, row in enumerate(rows):
        row_string = " | ".join([str(cell) for cell in row])
        table_string += f"row {i+1}: {row_string}\n"

    return table_string

def replace_df_from_list(input_string, tables):
    """
    Replace df<number> in the input string with the corresponding table name from the list.
    
    Args:
        input_string (str): The input string containing df<number>.
        tables (list): A list of table names.
    
    Returns:
        str: The modified string with df<number> replaced by the correct table name.
    """
    # Regex pattern to find df<number>
    pattern = r"df(\d+)"
    
    # Function to perform the replacement
    def replacer(match):
        # Extract the number from the match
        table_index = int(match.group(1)) - 1  # Convert to zero-based index
        if 0 <= table_index < len(tables):  # Ensure index is within bounds
            return tables[table_index]  # Replace with the corresponding table name
        return match.group(0)  # Return original if index is out of bounds
    
    # Perform the substitution
    return re.sub(pattern, replacer, input_string)

def replace_with_table_name(input_string, tables):
    """
    Replace T<number> in the input string with the corresponding table name from the list.
    
    Args:
        input_string (str): The input string containing T<number>.
        tables (list): A list of table names.
    
    Returns:
        str: The modified string with T<number> replaced by the correct table name.
    """
    # Regex pattern to find T<number>
    pattern = r"[Tt](\d+)"
    
    # Function to perform the replacement
    def replacer(match):
        # Extract the number from the match
        table_index = int(match.group(1)) - 1  # Convert to zero-based index
        if 0 <= table_index < len(tables):  # Ensure index is within bounds
            return tables[table_index]  # Replace with the corresponding table name
        return match.group(0)  # Return original if index is out of bounds
    
    # Perform the substitution
    return re.sub(pattern, replacer, input_string)


def generate_tables(predicted_table_ids, table_names, table_serialization_content, k=10):
    candidate_tables_string = []
    # get the top 10 table ids for each question
    for pred_ids in predicted_table_ids:
        entire_string = ""
        for table_id in pred_ids[:k]:
            table_name = table_names[table_id]
            columns, rows = table_serialization_content[table_id]
            table_string = create_table_string_representation(table_id, table_name, columns, rows)
            entire_string += table_string + "\n"
        candidate_tables_string.append(entire_string)

    return candidate_tables_string

def generate_tables_with_schema(predicted_table_ids, table_names, table_contents, k=10):
    candidate_tables_string = []
    # get the top k table ids for each question
    for pred_ids in predicted_table_ids:
        entire_string = ""
        for table_id in pred_ids[:k]:
            table_name = table_names[table_id]
            table_content = table_contents[table_id]
            columns = [col.lower() for col in table_content['columns']]
            table_string = f"table_id: {table_id}\ntable_name: {table_name}\ncolumns: {' | '.join(columns)}\n"
            entire_string += table_string + "\n"
        candidate_tables_string.append(entire_string)

    return candidate_tables_string

def generate_tables_random_rows(predicted_table_ids, table_names, table_contents, k=10):
    candidate_tables_string = []
    # get the top k table ids for each question
    for pred_ids in tqdm(predicted_table_ids):
        entire_string = ""
        for table_id in pred_ids[:k]:
            table_name = table_names[table_id]
            table_content = table_contents[table_id]
            # convert to df
            df = pd.DataFrame(table_content['data'], columns=table_content['columns'])
            # sample 2 rows
            rows = df.sample(min(5, len(df)), replace=False).values.tolist()
            columns = df.columns.tolist()
            rows = [[str(cell) for cell in row] for row in rows]
            table_string = create_table_string_representation(table_id, table_name, columns, rows)
            entire_string += table_string + "\n"
        candidate_tables_string.append(entire_string)

    return candidate_tables_string

def generate_tables_rel_rows(predicted_table_ids, table_docs, table_names, k=10):
    """
    Generate table string representations for the given predicted table IDs.
    
    Args:
        predicted_table_ids: List of predicted table IDs
        docs: List of document objects containing table data
        table_names: Dictionary mapping table IDs to table names
        k: Number of tables to process (default: 10)
    
    Returns:
        String containing formatted table representations
    """
    table_strings = []
    
    for table_id in predicted_table_ids[:k]:
        # Efficiently collect relevant documents
        temp_docs = []
        temp_docs.extend([json.loads(doc) for doc in table_docs[table_id][:5]])
                    
        if not temp_docs:  # Skip if no documents found
            continue
            
        # Create DataFrame and process data
        rows_df = pd.DataFrame(temp_docs)
        table_string = create_table_string_representation(
            table_id,
            table_names[table_id],
            rows_df.columns.tolist(),
            rows_df.astype(str).values.tolist()
        )
        table_strings.append(table_string)
    
    return "\n".join(table_strings)

def categorize_sql_columns(sql_columns):
    # Regex pattern to match nuanced SQL column names
    # Matches names with SQL functions, parentheses, or symbols
    nuanced_pattern = re.compile(r"[A-Z]+\([^)]*\)|\*|\(\)|\)")

    # Identify nuanced columns
    nuanced_columns = [
        col for col in sql_columns
        if nuanced_pattern.search(col)  # Matches SQL-style patterns
    ]

    # Identify basic column names (not nuanced)
    basic_columns = [
        col for col in sql_columns
        if col not in nuanced_columns
    ]

    return basic_columns, nuanced_columns

def build_schema_corpus(dfs):
    docs = []
    for ind, df in tqdm(enumerate(dfs), desc='Building schema corpus'):
        for col_name, col in df.items():
            if col.dtype != 'object' and col.dtype != str:
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}, "index": {ind}}}'
            else:
                most_freq_vals = col.value_counts().index.tolist()
                example_cells = most_freq_vals[:min(3, len(most_freq_vals))]
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "cell_examples": {example_cells}, "index": {ind}}}'
            docs.append(Document(page_content=col_name, metadata={'table_content': result_text}))
    print(f'Schema corpus built with {len(docs)} documents')
    return docs

def build_row_corpus(dfs):
    docs = []
    for ind, df in tqdm(enumerate(dfs), desc='Building row corpus...'):
        # Replace NaN, Infinity, and -Infinity with None for the entire DataFrame at once
        df_cleaned = df.replace([np.nan, np.inf, -np.inf], None)

        for idx, row in df_cleaned.iterrows():  # Use iterrows() to iterate through DataFrame rows
            # Convert the cleaned row to a dictionary
            row_dict = row.to_dict()

            # Use json.dumps() to ensure valid JSON formatting
            result_text = json.dumps(row_dict, ensure_ascii=False)  # No need for allow_nan=False

            # Append to document list
            docs.append(Document(page_content=result_text, metadata={'table_index': ind, 'row_index': idx+1}))

    print(f'Row corpus built with {len(docs)} documents')
    return docs


# function to get ground truth table ids
def get_ground_truth_table_ids(validation_data):
    """
    Get the ground truth table IDs from the validation data.
    
    Args:
        validation_data (list of dict): The validation data.
    
    Returns:
        list of list of int: The ground truth table IDs.
    """
    return [set(data["table_ids"]) for data in validation_data]

def infer_dtype(df):
    """
    Attempt to convert columns in a DataFrame to a more appropriate data type.

    :param df: Input DataFrame
    :return: DataFrame with updated dtypes
    """

    for col in df.columns:
        try:
            # Try converting to numeric
            df[col] = pd.to_numeric(df[col])

            # If the column type is still object (string) after trying numeric conversion, try datetime conversion
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col])
        except:
            pass

    return df

def convert_table_contents_to_dfs(table_contents):
        return [infer_dtype(pd.DataFrame(table_content['data'], columns=table_content['columns'])) for table_content in table_contents]