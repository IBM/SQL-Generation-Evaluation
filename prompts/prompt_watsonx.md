Your task is to convert a text question to a {db_type} query, given a database schema.
you must select one year column. select year_2022 column if No year is mentioned in user question.
Do not use count(*). select only one year column. 


The question that you must generate a SQL for is this `{user_question}`.

{instructions}

{table_metadata_string}
{k_shot_prompt}



Just return the SQL query, nothing else.
