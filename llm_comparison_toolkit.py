"""
Module: llm_comparison_toolkit

This module provides functionality for making API calls to Large Language Models (LLMs) using flexible prompt and system message templates. It includes classes and functions to manage rate limiting, create configuration dictionaries, make API calls from DataFrame rows, and compare different request configurations.

Classes:
- RateLimiter: Implements rate limiting functionality to ensure the number of actions (or 'tokens') does not exceed a specified maximum limit per minute.

Functions:
- create_config_dict_func: Creates a configuration dictionary for use with functions that make API calls to an LLM.
- make_api_call_from_dataframe_row: Makes an API call to an LLM using a configuration dictionary and a single row from a pandas DataFrame.
- use_df_to_call_llm_api: Performs multiple API calls to an LLM using a flexible prompt and system message template, saves the results as files in a subfolder of the 'corrected' folder.
- compare_request_configurations: Compares different prompt/system message/LLM configurations using a DataFrame as the basic data source.
- get_response_openai: Sends a prompt and system message to the OpenAI API, while managing the rate of API requests using a provided rate limiter.
- get_response_anthropic: Sends a prompt along with a system-generated message to the Anthropic API, managing the rate of requests with a provided rate limiter.

The module is designed to be flexible and agnostic to the specific LLM being used, allowing for testing a range of prompt formats and structures across different LLMs and facilitating comparisons among them.

Dependencies:
- openai
- pandas
- numpy
- os
- time
- tiktoken
- re
- anthropic
- collections
- inspect
- typing

Note: Calling an LLM requires an api key, this should be set up following the documentation for the default setting.
"""

import openai
#import config  # Import your config.py file this contains you openai api key
import pandas as pd
import numpy as np
import os
import time
import tiktoken
import re
from openai import OpenAI
import anthropic


import time
from collections import deque
import inspect
from typing import List, Callable, Optional


class RateLimiter:
    """
    A class to implement rate limiting functionality, ensuring that the number of actions (or 'tokens') 
    does not exceed a specified maximum limit per minute.

    The rate limiter uses a token bucket algorithm, represented by deques (double-ended queues), 
    to keep track of the number of tokens and their timestamps within the past minute.

    Attributes:
        max_tokens_per_minute (int): The maximum number of tokens that are allowed in one minute.
        tokens_deque (deque): A deque to store the number of tokens generated in the past minute.
        timestamps_deque (deque): A deque to store the timestamps of when tokens were generated in the past minute.

    Methods:
        add_tokens(tokens: int): Adds a specified number of tokens to the rate limiter. If adding the tokens 
        would exceed the maximum limit, the method pauses execution until it is permissible to add the tokens.

        check_tokens(tokens: int) -> bool: Checks if adding a specified number of tokens would exceed the 
        maximum limit, without actually adding them. Returns True if adding the tokens would stay within the 
        limit, and False otherwise.

    Example:
        rate_limiter = RateLimiter(100)  # Rate limiter allowing 100 tokens per minute
        rate_limiter.add_tokens(50)      # Add 50 tokens
        if rate_limiter.check_tokens(60): 
            rate_limiter.add_tokens(60)  # Add 60 tokens if it doesn't exceed the limit
    """
    def __init__(self, max_tokens_per_minute):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.tokens_deque = deque(maxlen=60) # Holds the tokens generated for the past minute.
        self.timestamps_deque = deque(maxlen=60) # Holds the timestamps of when tokens were generated.

    def add_tokens(self, tokens):
        current_time = time.time()

        # Removing tokens older than 1 minute
        while self.timestamps_deque and current_time - self.timestamps_deque[0] > 60:
            self.timestamps_deque.popleft()
            self.tokens_deque.popleft()

        # If the number of tokens is more than the maximum limit,
        # pause execution until it comes back down below the threshold
        if sum(self.tokens_deque) + tokens > self.max_tokens_per_minute:
            sleep_time = 60 - (current_time - self.timestamps_deque[0])
            time.sleep(sleep_time)

            # After sleeping, add the tokens and timestamps to the deque
            self.tokens_deque.append(tokens)
            self.timestamps_deque.append(current_time + sleep_time)
        else:
            # If the number of tokens is less than the maximum limit,
            # add the tokens and timestamps to the deque
            self.tokens_deque.append(tokens)
            self.timestamps_deque.append(current_time)

    def check_tokens(self, tokens):
        # Function to check if adding new tokens would exceed limit, without actually adding them
        current_time = time.time()
        while self.timestamps_deque and current_time - self.timestamps_deque[0] > 60:
            self.timestamps_deque.popleft()
            self.tokens_deque.popleft()

        return sum(self.tokens_deque) + tokens <= self.max_tokens_per_minute
    



def create_config_dict_func(
    get_response_func: Callable,
    rate_limiter: RateLimiter,
    engine: str,
    system_message_template: Optional[str] = None,
    prompt_template: str = "{text}", 
    additional_args: Optional[dict] = None
) -> dict:
    """
    Creates a configuration dictionary for use with functions that make API calls to a Language Model (LLM).
    
    This function initializes the configuration with essential keys and merges any additional arguments directly into the dictionary.

    Parameters:
    - get_response_func (Callable): A function from the llm_caller module that makes the call to the LLM.
    - rate_limiter (RateLimiter): A RateLimiter instance to prevent exceeding the LLM's rate limit.
    - system_message_template (str, optional): A string containing the message used as the system prompt. Defaults to None.
    - prompt_template (str): A string template for the prompt. Defaults to "{text}".
    - additional_args (dict, optional): Additional arguments to be directly merged into the configuration dictionary. Defaults to None.

    Note:
    There are other variables that are probably useful to have such as "engine", "max_tokens", 
    but these will have default variables in 'get_response_func'

    Returns:
    dict: A configuration dictionary with the specified settings, including any additional arguments.
    """
    config_dict = {
        "get_response_func": get_response_func,
        "rate_limiter": rate_limiter,
        "engine": engine,
        "system_message_template": system_message_template,
        "prompt_template": prompt_template
    }

    if additional_args:
        config_dict.update(additional_args)

    return config_dict

def make_api_call_from_dataframe_row(row, config_dict):

    """
    Makes an API call to a Large Language Model (LLM) using a configuration dictionary and a single row from a pandas DataFrame.

    This function is designed to be flexible and agnostic to the specific LLM being used. It leverages the configuration dictionary's "get_model_response_func" key to call a function specific to the LLM API framework, which should return a text string response.

    The function allows for testing a range of prompt formats and structures across different LLMs, facilitating comparisons among them.

    Parameters:
    - row (pd.Series): A single row from a pandas DataFrame that contains data to be used in the API call.
    - config_dict (dict): A configuration dictionary that must contain the following keys:
      - "prompt_template": A string template for the prompt to be sent to the LLM.
      - "system_message_template": A string template for system messages.
      - "get_model_response_func": A function to be called with the LLM, configured with keys from config_dict relevant to the function's parameters.

    Returns:
    str: A text string response from the LLM.

    The function constructs the prompt and system message by formatting the templates in the configuration dictionary with values from the specified DataFrame row. It then extracts the necessary arguments for the "get_model_response_func" function from the configuration dictionary and makes the API call to get the LLM response.
    """

    # Convert the row to a dictionary, filter keys based on the variables in the prompt template
    row_dict = {key: row[key] for key in re.findall(r'\{(.*?)\}', config_dict['prompt_template'])}
    # create the prompt as a minimum the prompt template must include the text to be recovered
    config_dict['prompt']  = config_dict['prompt_template'].format(**row_dict)

    # Convert the row to a dictionary, filter keys based on the variables in the system message
    row_dict = {key: row[key] for key in re.findall(r'\{(.*?)\}', config_dict['system_message_template'])}
    # Use ** to unpack the dictionary and .format() to insert the variables
    config_dict['system_message']  = config_dict['system_message_template'].format(**row_dict)

    #take out the model object
    get_response_func = config_dict['get_response_func']

    #extract the arguments from the function
    arguments = [param.name for param in inspect.signature(get_response_func).parameters.values()]

    #create new dictionary that only includes the arguments of the api caller function
    config_dict2 = {key: config_dict[key] for key in arguments if key in config_dict}

    #unpack the dictionary and unpack the dictionary in the function
    response = get_response_func(**config_dict2)
    return response


def use_df_to_call_llm_api(config_dict, df, response_name , folder_path='./data'):

    """
    Performs multiple api calls to an LLM using a flexible prompt and system message template, saves the results as files in a subfolder of the 
    'corrected' folder.
    The function is really supposed to be used to generate a large quantity of results in order to create comparison data between LLM's and prompts.

    Parameters:
    - config_dict (dict): A dictionary containing all the information required to all the api of an LLM and provide a system message and prompt.
        This dictionary is likely created by the 'create_config_dict_func' function.
    - df (pandas.DataFrame): DataFrame containing OCR data to be processed. Required columns are 'id'.
    - response_name (str): Descriptive name for the response, joined with the engine to create response folder.
    - engine (str, optional): The processing engine used for text recovery. Defaults to "gpt-3.5-turbo".
    - folder_path (str, optional): Project base data folder for recovered text, Defaults to './data'

    Process:
    - Checks for existing processed IDs to avoid reprocessing.
    - Calls api using make_api_call_from_dataframe_row, which returns a string
    - Save both the return string and the time taken to process to the corrected folder

    """
    # Create new subfolder path
    new_subfolder = os.path.join(folder_path, f"{response_name}_{config_dict['engine']}")
    if not os.path.exists(new_subfolder):
        os.makedirs(new_subfolder)

    # Path for the times_df CSV file
    times_csv_path = os.path.join(new_subfolder, f"0_processing_time.csv")

    # Check if times_df CSV file exists to continue from last save
    if os.path.exists(times_csv_path):
        times_df = pd.read_csv(times_csv_path)
    else:
        times_df = pd.DataFrame(columns=['id', 'time'])

    # Convert the 'id' column to a set for faster lookup
    processed_ids = set(times_df['id'])

    # List to accumulate new records
    #new_records = []

    for index, row in df.iterrows():
        if row['id'] not in processed_ids:
            start_time = time.time()  # Begin measuring time

            corrected_ocr = make_api_call_from_dataframe_row(row, config_dict) 

            end_time = time.time()  # Stop measuring time
            elapsed_time = round(end_time - start_time, 1)  # Time to the nearest tenth of a second

            # Construct file name and path
            file_path = os.path.join(new_subfolder, row['file_name'])
            
            # Save corrected_ocr as a text file
            with open(file_path, 'w') as file:
                file.write(corrected_ocr)
            
            # Append new record to the list
            new_records = {'id': row['id'], 'time': elapsed_time}

            new_record_df = pd.DataFrame([new_records], index = [0])
            
            if times_df.empty:
                times_df = new_record_df
            else:
                times_df = pd.concat([times_df, new_record_df], ignore_index=True)
            
            times_df.to_csv(times_csv_path, index=False)

            
            # Add the processed ID to the set
            processed_ids.add(row['id'])

def compare_request_configurations(df, configurations, folder_path = './data'):
    """
    Used to compare different prompt/system message/llm configurations using a dataframe as the basic datasource

    The function runs various configurations of prompt, system message and LLM engine and saves the results in a structured,
    folder. Is designed to help compare and tune approaches.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing base data to be processed.
    - configurations (list of dicts): A list where each dict contains the parameters for a
      `use_df_to_call_llm_api` call, see the function `create_config_dict_func` for how to create a configuration dictionary.
    """
    for config_dict in configurations:
        
        response_name = config_dict['response_name']

        # Call perform_capoc with the current configuration
        use_df_to_call_llm_api(config_dict, df, response_name , folder_path)


def get_response_openai(prompt, system_message, rate_limiter, engine="gpt-3.5-turbo", max_tokens = 4000, alt_endpoint = None):
    """
    Sends a prompt and system message to using the openAI library, while managing the rate of API requests 
    using a provided rate limiter.

    Parameters:
        prompt (str): The user's input text to be sent to the model.
        system_message (str): A system-generated message that precedes the user's prompt.
        rate_limiter (RateLimiter): An instance of the RateLimiter class to manage API request frequency.
        engine (str, optional): The model engine to use. Defaults to "gpt-3.5-turbo".
        alt_endpoint (dict, optional): If a non OpenAI model is being used such as Huggingface or Groq, use a dict as 
        {'base_url':"<ENDPOINT_URL>" + "/v1/", 'api_key':"<API_TOKEN>"}

    Returns:
        str: The trimmed content of the model's response if successful, None otherwise.

    Raises:
        openai.error.RateLimitError: If the rate limit for the API is exceeded.
        openai.error.APIError: For general API-related errors.
        openai.error.Timeout: If a timeout occurs while waiting for the API response.

    Example:
        rate_limiter = RateLimiter(100)
        response = get_model_response("Hello, world!", "System: Starting session", rate_limiter)
        print(response)
    """
    #create the encoding object, this allows us to acurrately find the total number of tokens and ensure we don't go over the rate limit
    #There may be a better way of doing this
    #enc = tiktoken.encoding_for_model(engine)
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo") #counting is all done using bytpair
    #print(alt_endpoint)
    if alt_endpoint:
  
        client =  OpenAI(
                base_url = alt_endpoint['base_url'], 
                api_key = alt_endpoint['api_key'],  
                )   
    else:    

        client = OpenAI() #default is to instantiate using open url endpoint and api key
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    attempts = 0
    while attempts < 5:
        try:
            prompt_length = len(enc.encode(prompt))  

            tokens = len(enc.encode(system_message)) + prompt_length

            #Overwrite the tokens if necessary, as the max tokens takes precedent
            if tokens> max_tokens:
                tokens = max_tokens
            
            # Add tokens to rate limiter and sleep if necessary
            rate_limiter.add_tokens(tokens)
                
            response = client.chat.completions.create(
                model = engine,
                messages = messages,
                max_tokens = max_tokens, 
                temperature = 0.2,
                top_p = 0.9,
            )

            return response.choices[0].message.content
            
        except openai.RateLimitError as e:
            print(f"RateLimitError encountered: {e}, waiting for a minute...")
            time.sleep(60)  # Wait for a minute before retrying
            continue  # Continue with the next iteration of the loop, thereby retrying the request
            
        except openai.APIError as e:
            print(f"APIError encountered: {e}, retrying in 5 seconds...")
            time.sleep(5)

        except openai.APITimeoutError as e:
            print(f"TimeoutError encountered: {e}, retrying in 10 seconds...")
            time.sleep(10)
            
        attempts += 1

    print("Failed to get model response after multiple attempts.")
    return None

def get_response_anthropic(prompt, system_message, rate_limiter, engine="claude-3-haiku-20240307", max_tokens = 4000):
    """
    Sends a prompt along with a system-generated message to the Anthropic API, managing the rate of requests 
    with a provided rate limiter. 

    Parameters:
        prompt (str): The user's input text to be sent to the model.
        system_message (str): A system-generated message that precedes the user's prompt.
        rate_limiter (RateLimiter): An instance of a custom RateLimiter class to manage the frequency of API requests.
        engine (str, optional): The specific model engine to use. Defaults to "claude-3-haiku-20240307".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4000.

    Returns:
        str: The model's response as a string, trimmed to remove unnecessary parts, if successful; None otherwise.

    Raises:
        anthropic.RateLimitError: If the rate limit for the API is exceeded.
        anthropic.APIError: For general API-related errors.
        Exception: For any other unexpected errors encountered during API interaction.

    Example:
        rate_limiter = RateLimiter(100)
        response = get_response_anthropic("Hello, world!", "System: Starting session", rate_limiter)
        print(response)
    """

    client = anthropic.Anthropic()

    attempts = 0
    while attempts < 5:
        try:
                
            response = client.messages.create(
                model=engine,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
                )
            #The response in and out tokens are added to the limiter
            rate_limiter.add_tokens(response.usage.input_tokens + response.usage.output_tokens)

            return response.content[0].text
            
        except anthropic.RateLimitError as e:
            print(f"Rate limit error: {e.message}")
            print("Pausing for 60 seconds before retrying...")
            time.sleep(60)
            # You can choose to retry the request here or handle it differently
        except anthropic.APIError as e:
            print(f"API error: {e.message}")

        except Exception as e:
            print(f"Unexpected error: {e}")
            # Handle any other unexpected errors
        else:
            # Process the successful response
            print("API request successful")
                
        attempts += 1

        print("Failed to get model response after multiple attempts.")
    return None



def generate_model_configs(model_configs_df, prompt_template, response_name_prefix):

    """
    Generates model configurations based on provided model configurations,
    a prompt template, and a response name prefix.

    Parameters:
        model_configs_df (DataFrame): DataFrame containing model configurations.
        prompt_template (str): Template string for prompts.
        response_name_prefix (str): Prefix for naming responses.

    Returns:
        list: List of dictionaries containing generated model configurations.

    Notes:
        This function iterates over each row in the model_configs DataFrame,
        updates the response name with the provided prefix, and creates a model configuration
        dictionary using the provided parameters.

    Example:
        model_configs_df = pd.DataFrame(...)
        prompt_template = "Enter your prompt here."
        response_name_prefix = "response"
        model_configs = generate_model_configs(model_configs_df, prompt_template, response_name_prefix)
    """
    model_configs = []
    for _, row in model_configs_df.iterrows():
        row['additional_args']['response_name'] = response_name_prefix +'_'
        model_configs.append(
            create_config_dict_func(
                get_response_func=row['get_response_func'],
                rate_limiter=RateLimiter(row['rate_limit']),
                engine=row['engine'],
                system_message_template="",
                prompt_template=prompt_template,
                additional_args=row['additional_args']
            )
        )
    return model_configs

