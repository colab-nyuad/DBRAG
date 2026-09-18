from typing import Dict, List, Optional, Callable
import pandas as pd
import json
import ast

# Constants
MODE_DESCRIPTIONS = {
    'all': "For each dataframe, here are the column names followed by all rows.\n\n",
    'first': "For each dataframe, here are the column names followed by the first few rows.\n\n",
    'random': "For each dataframe, here are the column names followed by some randomly sampled rows.\n\n",
    'relevant': "For each dataframe, here are the column names followed by some rows.\n\n",
}

def build_description(with_schema_only: bool = False, mode: str = 'first') -> str:
    """
    Build description string based on schema flag and retrieval mode.
    
    Args:
        with_schema_only: Whether only schema information is shown
        mode: Row selection mode ('first', 'random', 'relevant', 'all', 'summary')
        
    Returns:
        str: Appropriate description string
    """
    if with_schema_only:
        return "Here are the column names for each dataframe.\n"
        
    return MODE_DESCRIPTIONS.get(
        mode, 
        "For each dataframe, here are the column names followed by some sample rows.\n\n"
    )

def format_row(row: pd.Series) -> str:
    """Format a single row as a string."""
    return " | ".join(str(cell) for cell in row.values)

def get_schema_with_retrieved_cells(
    ranked_docs: Optional[Dict], 
    table_id: int, 
    table_df: pd.DataFrame,
    k: int = 2,
    mode: str = 'first'
) -> str:
    """
    Get the schema and selected rows of the table based on different strategies.
    
    Args:
        ranked_docs: Dictionary of ranked documents
        table_id: ID of the table
        table_df: DataFrame containing the table data
        k: Number of rows to retrieve
        mode: Selection mode for rows
        
    Returns:
        str: Formatted schema and row data
    """
    columns = ""
    rows = []

    if mode == 'all':
        columns = "columns : " + " | ".join(str(column) for column in table_df.columns)
        for index, row in table_df.iterrows():
            rows.append(f"row {index+1} : {format_row(row)}")
            
    elif mode == 'relevant' and ranked_docs and table_id in ranked_docs:
        try:
            docs = ranked_docs[table_id][:k]
            rows_df = pd.DataFrame(
                [json.loads(doc.page_content) for doc in docs],
                index=[doc.metadata['row_index'] for doc in docs]
            ).sort_index()
            
            columns = "columns : " + " | ".join(str(column) for column in rows_df.columns)
            for i, row in rows_df.iterrows():
                rows.append(f"row {i+1} : {format_row(row)}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing relevant rows: {str(e)}")
            
    elif mode == 'random':
        sampled_rows = table_df.sample(n=min(k, len(table_df)))
        columns = "columns : " + " | ".join(str(column) for column in sampled_rows.columns)
        for index, row in sampled_rows.iterrows():
            rows.append(f"row {index+1} : {format_row(row)}")
            
    else:  # 'first' mode
        sampled_rows = table_df.head(min(k, len(table_df)))
        columns = "columns : " + " | ".join(str(column) for column in sampled_rows.columns)
        for index, row in sampled_rows.iterrows():
            rows.append(f"row {index+1} : {format_row(row)}")
            
    return columns + "\n" + "\n".join(rows)

def build_table_from_retrieved_row_values(
    ranked_docs: Optional[Dict],
    table_names: List[str],
    table_ids: List[int],
    table_dfs: List[pd.DataFrame],
    k: int = 2,
    mode: str = 'first',
) -> str:
    """
    Build a string representation of the retrieved data from the tables.
    
    Args:
        ranked_docs: Dictionary of ranked documents
        table_ids: List of table IDs
        table_dfs: List of DataFrames
        k: Number of rows to retrieve
        mode: Selection mode for rows
        
    Returns:
        str: Formatted table data
    """
    final_str = build_description(with_schema_only=False, mode=mode)
    
    for i, tab_id in enumerate(table_ids):
        table_df = table_dfs[i]
        table_name = f"- df{i+1} -\ntable_name : {table_names[tab_id]}\n" if table_names else f"- df{i+1} -\n"
        
        if mode == 'schema':
            table_str = "columns : " + " | ".join(str(column) for column in table_df.columns)
        else:
            table_str = get_schema_with_retrieved_cells(
                ranked_docs,
                tab_id,
                table_df,
                k=k,
                mode=mode
            )
            
        final_str += table_name + table_str + "\n\n"
        
    return final_str.rstrip()

def create_filter_by_index(target_index: int) -> Callable:
    """Create a filter function for filtering by table index."""
    def filter_by_index(metadata: Dict) -> bool:
        if "table_content" not in metadata:
            return False
            
        try:
            table_content = ast.literal_eval(metadata["table_content"])
            return table_content.get("index") == target_index
        except (ValueError, SyntaxError):
            return False
            
    return filter_by_index

def build_table_from_retrieved_schema_values(
    query_embedding: List[float],
    table_names,
    table_ids: List[int],
    table_contents: Dict,
    k = 4,
) -> str:
    """
    Build a string representation of the retrieved schema data.
    
    Args:
        query_embedding: Vector embedding of the query
        table_ids: List of table IDs
        table_contents: Dictionary of table contents
        total_no_docs: Total number of documents to fetch
        
    Returns:
        str: Formatted schema data
    """
    final_str = build_description(with_schema_only=True)
    
    for i, table_id in enumerate(table_ids):
        table_name = f"- df{i+1} -\ntable_name : {table_names[tab_id]}\n" if table_names else f"- df{i+1} -\n"
       
        schema = "columns : " + " | ".join(table_contents[table_id]["columns"])
            
        final_str += table_name + schema + "\n\n"
        
    return final_str.rstrip()