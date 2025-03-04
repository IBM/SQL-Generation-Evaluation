import time
from typing import Dict, List
import os

from query_generators.query_generator import QueryGenerator
from utils.pruning import prune_metadata_str
from utils.gen_prompt import to_prompt_schema
from utils.dialects import convert_postgres_ddl_to_dialect
from ibm_watsonx_ai.foundation_models import Model
import tiktoken

class WatsonXQueryGenerator(QueryGenerator):
    """
    Query generator that uses IBM WatsonX.AI's models
    Models available may include language models optimized for database querying
    """

    def __init__(
        self,
        db_type: str,
        db_creds: Dict[str, str],
        db_name: str,
        model_id: str,
        #prompt_file: str,
        timeout: int,
        use_public_data: bool,
        verbose: bool,
        **kwargs,
    ):
        self.db_creds = db_creds
        self.db_type = db_type
        self.db_name = db_name
        self.model_id = model_id
        #self.prompt_file = prompt_file
        self.timeout = timeout
        self.use_public_data = use_public_data
        self.verbose = verbose
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("PROJECT_ID")
        self.generate_params = {"decoding_method": 'greedy',"max_new_tokens": 200,"GenParams.repetition_penalty":1, "GenParams.temperature":0}
        self.model = Model(model_id=self.model_id ,params=self.generate_params,credentials={"url" : "https://us-south.ml.cloud.ibm.com","apikey" : self.api_key},project_id=self.project_id)

    def get_completion(
        self,
        prompt,
        max_tokens=600,
        temperature=0,
        stop=None
    ):
        """Get completion using IBM WatsonX.AI model for a given prompt"""
        completion = ""
        try:
            completion = self.model.generate(
                prompt=prompt
            )
            
        except Exception as e:
            print('error'+str(e))
            if self.verbose:
                print(f"Error during text generation: {e}")
        return completion
    
    @staticmethod
    def count_tokens(
        model: str, messages: List[Dict[str, str]] = [], prompt: str = ""
    ) -> int:
        """
        This function counts the number of tokens used in a prompt
        model: the model used to generate the prompt. can be any valid OpenAI model
        messages: (only for OpenAI chat models) a list of messages to be used as a prompt. Each message is a dict with two keys: role and content
        """
        try:
            tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # default to o200k_base if the model is not in the list. this is just for approximating the max token count
            tokenizer = tiktoken.get_encoding("o200k_base")
        num_tokens = 0
        for message in messages:
            for _, value in message.items():
                num_tokens += len(tokenizer.encode(value))
        return num_tokens
    
    def generate_query(
        self,
        question: str,
        instructions: str,
        k_shot_prompt: str,
        glossary: str,
        table_metadata_string: str,
        prev_invalid_sql: str,
        prev_error_msg: str,
        table_aliases: str,
        columns_to_keep: int,
        shuffle: bool,
        prompt_file: str
    ) -> dict:
        start_time = time.time()
        self.err = ""
        self.query = ""
        self.reason = ""

        print('Publicdata',self.use_public_data)
        print('columns_to_keep',columns_to_keep)

        # Use public or private data based on configuration
        if self.use_public_data:
            from defog_data.metadata import dbs
            import defog_data.supplementary as sup
        else:
            from defog_data_private.metadata import dbs

        # Load and prepare prompt
        with open(prompt_file) as file:
            model_prompt = file.read()

        question_instructions = question + " " + instructions
        if table_metadata_string == "":
            if columns_to_keep > 0:
                pruned_metadata_ddl, join_str = prune_metadata_str(
                    question_instructions,
                    self.db_name,
                    self.use_public_data,
                    columns_to_keep,
                    shuffle,
                )
                pruned_metadata_ddl = convert_postgres_ddl_to_dialect(
                    postgres_ddl=pruned_metadata_ddl,
                    to_dialect=self.db_type,
                    db_name=self.db_name,
                )
                pruned_metadata_str = pruned_metadata_ddl + join_str
            elif columns_to_keep == 0:
                md = dbs[self.db_name]["table_metadata"]
                pruned_metadata_str = to_prompt_schema(md, shuffle)
                pruned_metadata_str = convert_postgres_ddl_to_dialect(
                    postgres_ddl=pruned_metadata_str,
                    to_dialect=self.db_type,
                    db_name=self.db_name,
                )
                column_join = sup.columns_join.get(self.db_name, {})
                # get join_str from column_join
                join_list = []
                for values in column_join.values():
                    col_1, col_2 = values[0]
                    # add to join_list
                    join_str = f"{col_1} can be joined with {col_2}"
                    if join_str not in join_list:
                        join_list.append(join_str)
                if len(join_list) > 0:
                    join_str = "\nHere is a list of joinable columns:\n" + "\n".join(
                        join_list
                    )
                else:
                    join_str = ""
                pruned_metadata_str = pruned_metadata_str + join_str
            else:
                raise ValueError("columns_to_keep must be >= 0")
        else:
            pruned_metadata_str = table_metadata_string        
        # metadata_str = self.prepare_metadata(
        #     question_instructions,
        #     columns_to_keep,
        #     shuffle,
        #     table_metadata_string
        # )
        #print(pruned_metadata_str)
        prompt = model_prompt.format(
            user_question=question,
            db_type=self.db_type,
            table_metadata_string=pruned_metadata_str,
            instructions=instructions,
            k_shot_prompt=k_shot_prompt,
            glossary=glossary,
            prev_invalid_sql=prev_invalid_sql,
            prev_error_msg=prev_error_msg,
            table_aliases=table_aliases,
        )

        try:
            self.completion = self.get_completion(
                prompt=prompt,
                max_tokens=200,
                temperature=0,
                stop=["```", ";"]
            )
            results = self.completion
            #print(prompt)
            print('results'+str(results))
            generated_text=results['results'][0]['generated_text']
            self.query = generated_text

            generated_text=generated_text.replace('sql','')
            if "```" in generated_text: 
                self.query = generated_text.split("```")[1]
                self.reason = "-"
            elif "`" in generated_text: 
                self.query=generated_text.split("`")[1]
                self.reason = "-"
        except Exception as e:
                print(e)
                self.handle_exception(e)
            

        return {
            "query": self.query,
            "reason": self.reason,
            "err": self.err,
            "latency_seconds": time.time() - start_time,
            "tokens_used": results['results'][0]['generated_token_count']+results['results'][0]['input_token_count'],
            "table_metadata_string":pruned_metadata_str,
            "prompt":prompt
        }

    # Additional methods for metadata preparation, error handling, and token counting would be similar to the Anthropic example

def handle_exception(self, e):
    if self.verbose:
        print(f"Error while generating query: {type(e).__name__}, {e}")
    self.query = ""
    self.reason = ""
    self.err = f"QUERY GENERATION ERROR: {type(e).__name__}, {e}"