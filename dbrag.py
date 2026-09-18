import argparse
import os
import pickle
import time
import warnings
import ast
import json
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Union, Annotated, TypedDict
import operator, logging
import numpy as np

import pandas as pd
import tiktoken
from tqdm import tqdm

# langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain import hub

# langgraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END

# local utilities (assumed to be available)
from utils.pickle_utils import load_pickle_file, save_as_pickle
from utils.prompt import (
    spider_prompt_with_selection,
    atis_prompt_with_selection,
    geoq_prompt_with_selection,
    spider_prompt_without_selection,
    atis_prompt_without_selection,
    geoq_prompt_without_selection,
)
from utils.dbrag_utils import (
    build_table_from_retrieved_row_values,
    build_table_from_retrieved_schema_values,
)
from utils.table_utils import infer_dtype
from QAEvaluator import MultiTableRAGEval

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

logging.basicConfig(
        filename="./run_logs/reader_errors.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w"
)

# Global variables for dataframes and for the row_db (used by some retrieval methods)
row_db = None

df_list=[]
def create_df_list(table_ids, table_contents):
    """Create list of dataframes from table_contents"""
    for tid in table_ids:
        columns = table_contents[tid]['columns']
        data = table_contents[tid]['data']
        df = infer_dtype(pd.DataFrame(data, columns=columns))
        df_list.append(df)

df_dic = {}
def create_df_dic(table_ids):
    """Create dictionary of dataframes for evaluation"""
    for i in range(len(table_ids)):
        df_dic[f"df{i + 1}"] = df_list[i]


def get_action(actions: str) -> str:
    """Extract the current action from the chain"""
    if "<BEGIN>" in actions:
        return actions.split('->')[1].strip()
    return actions.split('->')[0].strip()


def is_safe_pandas_code(code_str: str) -> tuple[bool, Optional[str]]:
    """Check if the pandas code is safe to execute"""
    try:
        tree = ast.parse(code_str)
        blocked_calls = {
            "to_csv", "to_excel", "to_json", "to_pickle", "to_sql",
            "plot", "to_string", "exec", "eval", "print", "display", "info"
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in blocked_calls:
                    return False, f"Unsafe operation detected: `{node.func.attr}()`"
                elif isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
                    return False, f"Unsafe operation detected: `{node.func.id}()`"
    except SyntaxError:
        return False, "Syntax error detected in the provided code"
    return True, None


@tool
def execute_dataframe_code(
    chain: Annotated[str, "The pandas chain of actions"],
    inter=None,
):
    """Execute the chain of pandas code on the dataframes"""
    name="execute_dataframe_code"
    try:
        action = get_action(chain)
        is_safe, error_message = is_safe_pandas_code(action)
        if not is_safe:
            return f"Error: {error_message}", action, None

        inter = eval(action, {"inter": inter, "df_dic": df_dic})
        if isinstance(inter, pd.DataFrame):
            intermediate = inter.head(70).to_markdown()
        else:
            intermediate = inter
        return intermediate, action, inter
    except Exception:
        return f"An exception occurred: {traceback.format_exc()}", action, None


@tool
def final_structured_output(
    columns: Annotated[Sequence[str], "The columns of the final inter table e.g ['name', 'age']"],
    rows: Annotated[Sequence[Sequence[Union[str, int, float, Decimal, None]]], "The rows of the final inter table e.g [['John', 25], ['Jane', 30]]"],
    code: Annotated[str, "The entire pandas code that generated the final inter table joined by ' -> ' e.g. df1.groupby('age').mean() -> df1.sort_values() -> <END>"],
    df_names: Annotated[Sequence[str], "The relevant dataframes out of the dataframes provided used to answer the question e.g. ['df1', 'df2']"]
):
    """Use this to return the final structured output of the table after the chain of actions"""
    name = "final_structured_output"
    return {"columns": list(columns), "rows": list(rows), "code": code, "df_names": list(df_names)}


def load_common_data(dataset: str, table_ids_path: str):
    """Load common data used by all methods"""
    base_path = f'./data/{dataset}'
    table_contents = load_pickle_file(f'{base_path}/table_contents.pkl')
    table_names = load_pickle_file(f'{base_path}/table_names.pkl')
    val_data = load_pickle_file(f'{base_path}/validation_data.pkl')
    query_embeddings = np.load(f'./retriever/outputs/openai_{dataset}_query_embeddings.npy')
    table_ids_list = np.load(table_ids_path).tolist()
    sql_code = load_pickle_file(f'{base_path}/sql_code.pkl')
    return table_contents, table_names, val_data, query_embeddings, table_ids_list, sql_code


# --- Retrieval functions for different methods ---

def get_retrieved_data_read_table(query_embedding, table_names, table_ids, total_docs, functions_and_input):
    """
    For the read_table methods, retrieve rows from each table.
    Uses the provided mode ("all" or "random") and number of rows (k).
    """
    return build_table_from_retrieved_row_values(
        {},
        table_names,
        table_ids,
        functions_and_input['df_list'],
        k=functions_and_input['k'],
        mode=functions_and_input['mode']
    )


def get_retrieved_data_schema_only(query_embedding, table_names, table_ids, total_docs, functions_and_input):
    return build_table_from_retrieved_schema_values(
        query_embedding,
        table_names,
        table_ids,
        functions_and_input['table_contents'],
        k=functions_and_input['k']
    )


def get_retrieved_data_schema_random(query_embedding, table_names, table_ids, total_docs, functions_and_input):
    return build_table_from_retrieved_row_values(
        {},
        table_names,
        table_ids,
        functions_and_input['df_list'],
        k=functions_and_input['k'],
        mode="random"
    )


def get_retrieved_data_schema_relevant(query_embedding, table_names,  table_ids, total_docs, functions_and_input):
    global row_db
    docs = row_db.similarity_search_by_vector(query_embedding, k=total_docs)
    table_docs = defaultdict(list)
    for doc in docs:
        table_docs[doc.metadata["table_index"]].append(doc)
    return build_table_from_retrieved_row_values(
        table_docs,
        table_names,
        table_ids,
        functions_and_input['df_list'],
        k=functions_and_input['k'],
        mode="relevant"
    )


def get_total_docs_default():
    return 0

def get_total_docs_schema_relevant():
    global row_db
    return row_db.index.ntotal


# --- Graph and workflow helper functions ---

def define_graph_cycles(model, tool_executor):
    """Define the graph cycles for the agent."""
    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        return "continue" if "function_call" in last_message.additional_kwargs else "end"

    def call_model(state):
        response = model.invoke(state)
        return {"messages": [response]}

    def call_tool(state):
        messages = state['messages']
        last_message = messages[-1]
        tool_input = last_message.additional_kwargs["function_call"]["arguments"]
        tool_input_dict = json.loads(tool_input)
        tool_input_dict['inter'] = state['inter']
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=tool_input_dict
        )
        tool_name = action.tool
        response = tool_executor.invoke(action)
        if tool_name == 'execute_dataframe_code':
            response, attempted_action, inter = response
            if "An exception occurred:" in str(response):
                error_info = f"""
                You have previously performed the actions: 
                {state['actions']}

                Current action: 
                {attempted_action}

                Result .head(50): 
                {response}

                You must correct your approach and continue until you can answer the question:
                {state['question']}

                Continue the chain with the following format: action_i -> action_i+1 ... -> <END>
                """
                return {"messages": [FunctionMessage(content=error_info, name=tool_name)]}
            success_info = f"""
            You have previously performed the actions: 
            {state['actions']}

            Current action: 
            {attempted_action}

            Result .head(50):
            {response}

            You must continue until you can answer the question:
            {state['question']}

            Continue the  chain with the following format: action_i -> action_i+1 ... -> <END>
            """
            return {
                "messages": [FunctionMessage(content=success_info, name=tool_name)],
                "actions": [attempted_action],
                "inter": inter
            }
        return {
            "messages": [FunctionMessage(content=str(response), name=tool_name)],
            "final_table": response if tool_name == 'final_structured_output' else None
        }
    return call_model, call_tool, should_continue

# create graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[str], operator.add]
    inter: pd.DataFrame
    question: str
    memory: str
    final_table: dict

def create_workflow(call_model, call_tool, should_continue):
    """Create the agent workflow."""
    # Note: The StateGraph uses AgentState as its type.
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
    workflow.add_edge('action', 'agent')
    return workflow.compile()


# --- Main processing functions ---

def process_question(i: int, data: dict, total_docs: int, functions_and_input: dict):
    """
    Process a single question. This resets the global dataframes,
    creates the tool executor, sets up the prompt, and executes the workflow.
    """
    global df_dic, df_list
    df_dic = {}
    df_list = []

    tools = [execute_dataframe_code, final_structured_output]
    tool_executor = ToolExecutor(tools)
    functions = [convert_to_openai_function(t) for t in tools]

    # Setup dataframes
    table_ids = functions_and_input['table_ids_list'][i] if not functions_and_input['use_gt_tab_ids'] else data['table_ids']
    table_names = [] if functions_and_input['dataset'] == 'spider' else functions_and_input['table_names']
    create_df_list(table_ids, functions_and_input['table_contents'])
    create_df_dic(table_ids)
    functions_and_input['df_list'] = df_list

    # Get retrieved data (using the appropriate retrieval function)
    retrieved_data_str = functions_and_input['get_retrieved_data'](functions_and_input['query_embeddings'][i], table_names, table_ids, total_docs, functions_and_input)

    # Create prompt (using langchain’s ChatPromptTemplate)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", functions_and_input['prompt']),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt_template = prompt_template.partial(num_dfs=len(df_list))
    prompt_template = prompt_template.partial(tool_names=", ".join([t.name for t in tools]))
    prompt_template = prompt_template.partial(retrieved_data=retrieved_data_str)

    text_prompt = prompt_template.format(messages=[])
    tokenized_prompt = functions_and_input['tokenizer'].encode(text_prompt)
    if len(tokenized_prompt) > 16385:
        return i, {
            "question": data["question"],
            "gold_answer": data["answer"],
            "predicted_answer": {"columns": [], "data": []},
            "code": "",
            "df_names": []
        }, "Prompt too long"

    model = prompt_template | ChatOpenAI(model=functions_and_input['model'], temperature=0.8).bind_functions(functions)
    call_model, call_tool, should_continue = define_graph_cycles(model, tool_executor)
    app = create_workflow(call_model, call_tool, should_continue)

    try:
        inputs = {
            "messages": [HumanMessage(content=data['question'])],
            "actions": ["<BEGIN>"],
            "question": data['question'],
            "memory": "",
            "inter": None
        }
        for output in app.stream(inputs, {"recursion_limit": 60}):
            for key, value in output.items():
                if key == "action" and value and value['messages']:
                    # Note: now final_structured_output returns a dict
                    if value["messages"][0].name == "final_structured_output":
                        final_table = value["final_table"]
                        return i, {
                            "question": data["question"],
                            "gold_answer": data["answer"],
                            "predicted_answer": {"columns": final_table["columns"], "data": final_table["rows"]},
                            "code": final_table["code"],
                            "df_names": final_table["df_names"]
                        }, None
        return i, {
            "question": data["question"],
            "gold_answer": data["answer"],
            "predicted_answer": {"columns": [], "data": []},
            "code": "",
            "df_names": []
        }, "No final output"
    except Exception as e:
        # print(f"Error processing question {i}: {e}")
        # log the error
        logging.error(f"Error processing question {i}: {e}")
        return i, {
            "question": data["question"],
            "gold_answer": data["answer"],
            "predicted_answer": {"columns": [], "data": []},
            "code": "",
            "df_names": []
        }, str(e)


def process_all_data(functions_and_input: dict, batch_size: int, sc: int, max_workers: int):
    """
    Process all validation questions in batches.
    Returns two dictionaries: one with answers and one with any errors.
    """
    final_table_answers = defaultdict(lambda: {'question': '', 'gold_answer': {}, 'predicted_answers': [], 'codes': [], 'df_names': []})
    errors = defaultdict(list)
    starting_index = 0
    st = 0
    print("Starting fresh run")

    num_batches = (len(functions_and_input['val_data']) + batch_size - 1) // batch_size
    total_docs = functions_and_input['get_total_docs']()


    with tqdm(total=(len(functions_and_input['val_data']) - st) * sc, desc="Processing questions") as overall_pbar:
        for batch_idx in range(starting_index // batch_size, num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(functions_and_input['val_data']))
            batch_data = functions_and_input['val_data'][start_idx:end_idx]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_question, i, data, total_docs, functions_and_input): i
                    for i, data in enumerate(batch_data, start=start_idx)
                    for _ in range(sc)
                }
                for future in as_completed(futures):
                    idx, result, error = future.result(timeout=60)
                    if final_table_answers[idx]['question'] == '' and final_table_answers[idx]['gold_answer'] == {}:
                        final_table_answers[idx]['question'] = result['question']
                        final_table_answers[idx]['gold_answer'] = result['gold_answer']
                    final_table_answers[idx]['predicted_answers'].append(result['predicted_answer'])
                    final_table_answers[idx]['codes'].append(result['code'])
                    final_table_answers[idx]['df_names'].append(result['df_names'])
                    if error:
                        errors[idx].append(error)
                    overall_pbar.update(1)
    return dict(final_table_answers), dict(errors)


def vote_for_consistency(answers: dict) -> dict:
    """
    For each question, vote among the multiple predicted answers for consistency.
    The most common answer is selected as the final answer.
    """
    for idx, details in answers.items():
        filtered_answers = [
            pred for pred in details['predicted_answers']
            if pred['columns'] and pred['data']
        ]
        if not filtered_answers:
            details['final_answer'] = {"columns": [], "data": []}
            details['associated_code'] = []
            details['associated_df_names'] = []
            continue

        answer_counter = Counter(
            (tuple(pred['columns']), tuple(tuple(row) for row in pred['data']))
            for pred in filtered_answers
        )
        most_common_answer, _ = answer_counter.most_common(1)[0]
        final_answer = {
            "columns": list(most_common_answer[0]),
            "data": [list(row) for row in most_common_answer[1]]
        }
        for pred, code, df_names in zip(
            details['predicted_answers'],
            details['codes'],
            details['df_names']
        ):
            if (pred['columns'] == final_answer['columns'] and 
                pred['data'] == final_answer['data']):
                details['associated_code'] = code
                details['associated_df_names'] = df_names
                break
        else:
            details['associated_code'] = []
            details['associated_df_names'] = []
        details['final_answer'] = final_answer
    return answers


def save_results(answers, errors, method_name: str, output_dir: str, model: str):
    """Save processing results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{method_name}_{model}"
    save_as_pickle(answers, f'{output_dir}/{base_filename}.pkl')
    save_as_pickle(errors, f'{output_dir}/{base_filename}_errors.pkl')


def run_rag_evaluation(answers, val_data, sql_code, dataset_name, reader_method, method, output_dir, truncate=False, with_columns=True):
    """
    Initialize and run MultiTableRAGEval and return the evaluation results.
    """
    rag_evaluator = MultiTableRAGEval(dataset_name, reader_method, method, output_dir, truncate=truncate, with_column_names=with_columns)
    results = rag_evaluator.evaluate(answers, val_data, sql_code)
    rag_evaluator.save_results(results)
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process data using different methods')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model name (e.g., gpt-3.5-turbo)')
    parser.add_argument('--dataset', type=str, required=True, choices=['spider', 'atis', 'geoq'], help='Dataset name')
    parser.add_argument('--method', type=str, required=True,
                        choices=['read_all', 'read_k', 'schema_only', 'schema_random', 'schema_relevant'],
                        help='Method to use')
    parser.add_argument('--table_ids_path', type=str, required=True , help='Path to reranked table ids for the dataset')
    parser.add_argument('--output_dir', type=str, default='./reader/outputs', help='Output directory')
    parser.add_argument('--eval_dir', type=str, default='./reader/evaluation', help='Evaluation directory')
    parser.add_argument('--k', type=int, default=2, help='Number of rows for random and relevant methods')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--sc', type=int, default=5, help='Number of samples per question(self-consistency)')
    parser.add_argument('--max_workers', type=int, default=50, help='Maximum number of workers')
    parser.add_argument('--use_gt_tab_ids', action='store_true', help='Use ground truth table ids')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Build a full state dictionary for passing to the processing functions.
    functions_and_input = {}
    functions_and_input['model'] = args.model
    # Choose prompt based on method.
    if args.method in ['read_all', 'read_k']:
        functions_and_input['prompt'] = spider_prompt_without_selection if args.dataset == 'spider' else atis_prompt_without_selection if args.dataset == 'atis' else geoq_prompt_without_selection
    else:
        functions_and_input['prompt'] = spider_prompt_with_selection if args.dataset == 'spider' else atis_prompt_with_selection if args.dataset == 'atis' else geoq_prompt_with_selection
    functions_and_input['dataset'] = args.dataset
    functions_and_input['table_ids'] = args.table_ids_path
    functions_and_input['output_dir'] = args.output_dir
    functions_and_input['k'] = args.k
    functions_and_input['use_gt_tab_ids'] = args.use_gt_tab_ids

    # Set up method-specific parameters.
    if args.method == 'read_all':
        functions_and_input['mode'] = "all"
        functions_and_input['get_retrieved_data'] = get_retrieved_data_read_table
        functions_and_input['get_total_docs'] = get_total_docs_default
    elif args.method == 'schema_only':
        functions_and_input['get_retrieved_data'] = get_retrieved_data_schema_only   
        functions_and_input['get_total_docs'] = get_total_docs_default
    elif args.method == 'schema_random':
        functions_and_input['get_retrieved_data'] = get_retrieved_data_schema_random
        functions_and_input['get_total_docs'] = get_total_docs_default
    elif args.method == 'schema_relevant' or args.method=='read_k':
        global row_db
        row_db = FAISS.load_local(
            f"./reranker/outputs/{args.dataset}_rowdb",
            OpenAIEmbeddings(model='text-embedding-3-small'),
            allow_dangerous_deserialization=True
        )
        functions_and_input['get_retrieved_data'] = get_retrieved_data_schema_relevant 
        functions_and_input['get_total_docs'] = get_total_docs_schema_relevant

    # Load common data.
    (functions_and_input['table_contents'], functions_and_input['table_names'], functions_and_input['val_data'], 
    functions_and_input['query_embeddings'], functions_and_input['table_ids_list'], functions_and_input['sql_code']) = load_common_data(args.dataset, args.table_ids_path)

    # Initialize the tokenizer.
    functions_and_input['tokenizer'] = tiktoken.encoding_for_model(args.model)

    # Process the data.
    answers, errors = process_all_data(functions_and_input, args.batch_size, args.sc, args.max_workers)
    save_results(answers, errors, args.method, args.output_dir, args.model)

    # Vote for consistency among samples.
    answers_with_voting = vote_for_consistency(answers)
    save_results(answers_with_voting, errors, f"{args.dataset}_{args.method}_consistency", args.output_dir, args.model)

    os.makedirs(args.eval_dir, exist_ok=True)
    results_with_columns = run_rag_evaluation(
        answers_with_voting,
        functions_and_input['val_data'],
        functions_and_input['sql_code'],
        args.dataset,
        args.method,
        "mtrag_with_columns",
        args.eval_dir,
        with_columns=True
    )
    results_without_columns = run_rag_evaluation(
        answers_with_voting,
        functions_and_input['val_data'],
        functions_and_input['sql_code'],
        args.dataset,
        args.method,
        "mtrag_no_columns",
        args.eval_dir,
        with_columns=False
    )

    unanswered_q = sum(1 for k, v in errors.items() if len(v) == args.sc)
    print(f"Number of unanswered questions: {unanswered_q}")
    print(f"Number of answered questions: {len(answers) - unanswered_q}")


if __name__ == "__main__":
    main()
