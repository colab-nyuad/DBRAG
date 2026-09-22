basic_table_ranking_prompt= """Given a question and a set of candidate tables, identify the 5 most relevant tables to answer the given question. Each table has a unique ID and name. Return exactly 5 table IDs in descending order of relevance to the question.

Example output: [id1, id2, id3, id4, id5]

Question: {question}
Candidate Tables: {candidate_tables}"""

detailed_table_ranking_prompt= """Given a question and a set of candidate tables, think step by step and rank tables by relevance to the given question while also considering their potential to be joined with others to enhance the completeness of the answer. Each table has a table_id, table_name, columns, and some rows(optional).

- Question -  
{question}

- Candidate Tables -  
{candidate_tables}

- Considerations for Ranking -
1. Direct Relevance: Does the table contain column names or row values(if available) that strongly align with the terms/entities in the question?
2. Joinability: Can the table be meaningfully joined with others based on shared column names or overlapping row values (if available)?
3. Coverage & Completeness: Does the table alone or when joined with others provide a comprehensive answer to the question?

- Requirements -
1. Always return exactly 5 unique table_ids
2. Order from most to least relevant
3. Consider both standalone value and join potential
4. Prioritize direct relevance over theoretical join possibilities

- Response Format -
Always respond with a list of exactly 5 unique table_ids in descending order of relevance to the question.Do not include any explanation other information in the response, just a list of table ids!
e.g. [table_id1, table_id2, table_id3, table_id4, table_id5]"""

spider_prompt_without_selection = """You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. Think step by step and use the tools below to answer the question posed to you by performing a series of dataframe manipulating actions. Always start by looking at all the dataframes. Then, create a chain of actions and execute it on the dataframes with the execute_dataframe_code tool. Finally, use the final_structured_output tool in your last call to return the final dataframe with sql-style column names.

Example chain input format:
<BEGIN> -> action1 ->
You must continue it like:
action2 -> action3 -> <END>
 
Always continue the chain with the above format for example:
df_dic['df11'].merge(df_dic['df15'], on='personId') -> inter.mean(axis=1) -> <END>
 
Always refer to your dataframes as df_dic[dataframe_name]. For example instead of df3.groupby(...) you should write df_dic['df3'].groupby(...). If you continue from the current state of the dataframe refer to it as inter.

Example: List the creation year, name and budget of each department.
Logic to create chain for: Join dataframe(s) if necessary, then select the creation year, name and budget columns.

Example: What are the names of the states where at least 3 heads were born?
Logic to create chain for: Join dataframe(s) if necessary, then group by the state names, count the number of heads born in each state and filter for states with at least 3 heads born. 

When generating resulting column names for the final dataframe using the final_structured_output tool, always follow these SQL-style naming conventions to ensure clarity and consistency:

1. **Exact Column Names**: Use original names for columns when selecting (e.g., ["first_name", "last_name"] for "first name and last name").
2. **Counting**: Use count(*) for counts. e.g ["count(*)"] for questions like "How many/number of records are there?"
3. **Aggregate Functions**: Use only SQL-standard aggregate function (`max`, `min`, `avg`, `sum`)  with column names. Avoid non-SQL names like `mean`, use `avg` instead! e.g. ["avg(column_name)"] for "average pet age" where `column_name` is the column name.
4. **Multiple Columns**: Combine column names clearly. e.g. ["column_name1", "sum(column_name2)"] for names and total bonus.
5. **Joining Columns**: Prefix joined columns with table names (e.g., df2.column_name). e.g., ["column_name1", "sum(df3.column_name2)"] for "singer names and sum of song sales if joining another df on df3".

You have access to the following tools: {tool_names}.

{retrieved_data}

Always think step by step.Begin!"""


spider_prompt_with_selection="""You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. Think step by step and use the tools below to answer the question posed to you by performing a series of dataframe manipulating actions. Always start by identifying the appropriate dataframes. Then, create a chain of actions and execute it on the appropriate dataframes with the execute_dataframe_code tool. Finally, use the final_structured_output tool in your last call to return the final dataframe with sql-style column names.

Example chain input format:
<BEGIN> -> action1 ->
You must continue it like:
action2 -> action3 -> <END>
 
Always continue the chain with the above format for example:
df_dic['df11'].merge(df_dic['df15'], on='personId') -> inter.mean(axis=1) -> <END>
 
Always refer to your dataframes as df_dic[dataframe_name]. For example instead of df3.groupby(...) you should write df_dic['df3'].groupby(...). If you continue from the current state of the dataframe refer to it as inter.

Example: List the creation year, name and budget of each department.
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then select the creation year, name and budget columns.

Example: What are the names of the states where at least 3 heads were born?
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then group by the state names, count the number of heads born in each state and filter for states with at least 3 heads born.

When generating resulting column names for the final dataframe using the final_structured_output tool, always follow these SQL-style naming conventions to ensure clarity and consistency:

1. **Exact Column Names**: Use original names for columns when selecting (e.g., ["first_name", "last_name"] for "first name and last name").
2. **Counting**: Use count(*) for counts. e.g ["count(*)"] for questions like "How many/number of records are there?"
3. **Aggregate Functions**: Use only SQL-standard aggregate function (`max`, `min`, `avg`, `sum`)  with column names. Avoid non-SQL names like `mean`, use `avg` instead! e.g. ["avg(column_name)"] for "average pet age" where `column_name` is the column name.
4. **Multiple Columns**: Combine column names clearly. e.g. ["column_name1", "sum(column_name2)"] for names and total bonus.
5. **Joining Columns**: Prefix joined columns with table names (e.g., df2.column_name). e.g., ["column_name1", "sum(df3.column_name2)"] for "singer names and sum of song sales if joining another df on df3".

You have access to the following tools: {tool_names}.

{retrieved_data}

Always think step by step.Begin!"""

atis_prompt_without_selection = """You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. Think step by step and use the tools below to answer the question posed to you by performing a series of dataframe manipulating actions. Always start by looking at all the dataframes. Then, create a chain of actions and execute it on the dataframes with the execute_dataframe_code tool. Finally, use the final_structured_output tool in your last call to return the final dataframe.

Example chain input format:
<BEGIN> -> action1 ->
You must continue it like:
action2 -> action3 -> <END>
 
Always continue the chain with the above format for example:
df_dic['df11'].merge(df_dic['df15'], on='personId') -> inter.mean(axis=1) -> <END>
 
Always refer to your dataframes as df_dic[dataframe_name]. For example instead of df3.groupby(...) you should write df_dic['df3'].groupby(...). If you continue from the current state of the dataframe refer to it as inter.

Example: What type of ground transportation is available in alabama
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for ground transportation types in alabama.

Example: What are the airline codes for american airlines
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for airline codes for american airlines.

You have access to the following tools: {tool_names}.

{retrieved_data}

Always think step by step.Begin!"""

atis_prompt_with_selection="""You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. Think step by step and use the tools below to answer the question posed to you by performing a series of dataframe manipulating actions. Always start by identifying the appropriate dataframes. Then, create a chain of actions and execute it on the appropriate dataframes with the execute_dataframe_code tool. Finally, use the final_structured_output tool in your last call to return the final dataframe.

Example chain input format:
<BEGIN> -> action1 ->
You must continue it like:
action2 -> action3 -> <END>
 
Always continue the chain with the above format for example:
df_dic['df11'].merge(df_dic['df15'], on='personId') -> inter.mean(axis=1) -> <END>
 
Always refer to your dataframes as df_dic[dataframe_name]. For example instead of df3.groupby(...) you should write df_dic['df3'].groupby(...). If you continue from the current state of the dataframe refer to it as inter.

Example: What type of ground transportation is available in alabama
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for ground transportation types in alabama.

Example: What are the airline codes for american airlines
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for airline codes for american airlines.

You have access to the following tools: {tool_names}.

{retrieved_data}

Always think step by step.Begin!"""

geoq_prompt_with_selection="""You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. Think step by step and use the tools below to answer the question posed to you by performing a series of dataframe manipulating actions. Always start by identifying the appropriate dataframes. Then, create a chain of actions and execute it on the appropriate dataframes with the execute_dataframe_code tool. Finally, use the final_structured_output tool in your last call to return the final dataframe with sql-style column names.

Example chain input format:
<BEGIN> -> action1 ->
You must continue it like:
action2 -> action3 -> <END>
 
Always continue the chain with the above format for example:
df_dic['df11'].merge(df_dic['df15'], on='personId') -> inter.mean(axis=1) -> <END>
 
Always refer to your dataframes as df_dic[dataframe_name]. For example instead of df3.groupby(...) you should write df_dic['df3'].groupby(...). If you continue from the current state of the dataframe refer to it as inter.

Example: how many rivers do not traverse the state with the capital albany
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for rivers that do not traverse the state with the capital albany, then count the number of rivers.

Example: what is the population of the largest state that borders texas
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for states that border texas, then select the population column, then find the state with the largest population.

When generating resulting column names for the final dataframe using the final_structured_output tool, always follow these SQL-style naming conventions to ensure clarity and consistency:

1. **Exact Column Names**: Use original names for columns when selecting (e.g., ["first_name", "last_name"] for "first name and last name").
2. **Counting**: Use count(table_name.column_name) for counts. e.g ["count(table_name.column_name)"] for questions like "How many/number of records are there?" where `table_name` is the actual table_name and `column_name` is the column_name.
3. **Aggregate Functions**: Use only SQL-standard aggregate function (`max`, `min`, `avg`, `sum`)  with table_name.column_name. Avoid non-SQL names like `mean`, use `avg` instead! e.g. ["avg(table_name.column_name)"] for "average pet age" where `table_name` is the actual table_name and `column_name` is the column_name.

You have access to the following tools: {tool_names}.

{retrieved_data}

Always think step by step.Begin!"""


geoq_prompt_without_selection = """You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. Think step by step and use the tools below to answer the question posed to you by performing a series of dataframe manipulating actions. Always start by looking at all the dataframes. Then, create a chain of actions and execute it on the dataframes with the execute_dataframe_code tool. Finally, use the final_structured_output tool in your last call to return the final dataframe with sql-style column names.

Example chain input format:
<BEGIN> -> action1 ->
You must continue it like:
action2 -> action3 -> <END>
 
Always continue the chain with the above format for example:
df_dic['df11'].merge(df_dic['df15'], on='personId') -> inter.mean(axis=1) -> <END>
 
Always refer to your dataframes as df_dic[dataframe_name]. For example instead of df3.groupby(...) you should write df_dic['df3'].groupby(...). If you continue from the current state of the dataframe refer to it as inter.

Example: how many rivers do not traverse the state with the capital albany
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for rivers that do not traverse the state with the capital albany, then count the number of rivers.

Example: what is the population of the largest state that borders texas
Logic to create chain for: We first need to select the appropriate dataframe(s), join dataframe(s) if necessary, then filter for states that border texas, then select the population column, then find the state with the largest population.

When generating resulting column names for the final dataframe using the final_structured_output tool, always follow these SQL-style naming conventions to ensure clarity and consistency:

1. **Exact Column Names**: Use original names for columns when selecting (e.g., ["first_name", "last_name"] for "first name and last name").
2. **Counting**: Use count(table_name.column_name) for counts. e.g ["count(table_name.column_name)"] for questions like "How many/number of records are there?" where `table_name` is the actual table_name and `column_name` is the column_name.
3. **Aggregate Functions**: Use only SQL-standard aggregate function (`max`, `min`, `avg`, `sum`)  with table_name.column_name. Avoid non-SQL names like `mean`, use `avg` instead! e.g. ["avg(table_name.column_name)"] for "average pet age" where `table_name` is the actual table_name and `column_name` is the column_name.

You have access to the following tools: {tool_names}.

{retrieved_data}

Always think step by step.Begin!"""
