import os
import requests
from bs4 import BeautifulSoup
import tiktoken
import pandas as pd

##
## These are function that are designed to support the processing of the overproof dataset
##
## This is only a small part of the project so not much has been done on it
##

def save_processed_data_line(processed_data, output_dir):

    """
    Saves processed data with line-wise division into raw and corrected folders.

    Parameters:
        processed_data (list): List of dictionaries containing processed data.
        output_dir (str): Directory path where the processed data will be saved.

    Notes:
        This function should only be run on sampled data as it divides the data
        into raw and corrected lines and saves them separately.

    Example:
        save_processed_data_line(processed_data, '/path/to/output_directory')
    """
    raw_dir = os.path.join(output_dir, 'line_raw')
    corrected_dir = os.path.join(output_dir, 'line_corrected')

    # Create the output directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(corrected_dir, exist_ok=True)

    for sub_doc in processed_data:
        metadata = sub_doc['metadata']
        raw_lines = sub_doc['raw'].split('\n')
        corrected_lines = sub_doc['corrected'].split('\n')

        # Replace spaces with underscores in the metadata string
        metadata_file_name = metadata.replace(' ', '_')

        for i, (raw_line, corrected_line) in enumerate(zip(raw_lines, corrected_lines), start=1):
            # Create file names with line numbers
            raw_file_name = f"{metadata_file_name}_line{i}.txt"
            corrected_file_name = f"{metadata_file_name}_line{i}.txt"

            # Save raw line to the 'raw' folder
            raw_file_path = os.path.join(raw_dir, raw_file_name)
            with open(raw_file_path, 'w', encoding='utf-8') as raw_file:
                raw_file.write(raw_line)

            # Save corrected line to the 'corrected' folder
            corrected_file_path = os.path.join(corrected_dir, corrected_file_name)
            with open(corrected_file_path, 'w', encoding='utf-8') as corrected_file:
                corrected_file.write(corrected_line)


def save_processed_data(processed_data, output_dir, keys):
    """
    Saves processed data with specified keys into separate folders.

    Parameters:
        processed_data (list): List of dictionaries containing processed data.
        output_dir (str): Directory path where the processed data will be saved.
        keys (list): List of keys to be saved into separate folders.

    Example:
        save_processed_data(processed_data, '/path/to/output_directory', ['raw', 'corrected'])
    """
    for key in keys:
        key_dir = os.path.join(output_dir, key)
        os.makedirs(key_dir, exist_ok=True)

    for sub_doc in processed_data:
        metadata = sub_doc['metadata']
        file_name = metadata.replace(' ', '_') + '.txt'

        for key in keys:
            if key in sub_doc:
                key_dir = os.path.join(output_dir, key)
                file_path = os.path.join(key_dir, file_name)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(sub_doc[key])


def process_txt_file(file_path, column_names):
    """
    Process a text file and extract sub-documents based on the specified column names.

    Args:
        file_path (str): The path to the text file.
        column_names (list): A list of column names representing the parts of each line separated by '||@@||'.

    Returns:
        list: A list of dictionaries, where each dictionary represents a sub-document with the following keys:
            - 'metadata': The metadata of the sub-document.
            - column_name: The concatenated values for each specified column name.

    Example:
        # For files with 'raw' and 'corrected' columns
        result_d1 = process_txt_file('file_path_d1.txt', ['raw', 'corrected'])

        # For files with 'raw', 'corrected', and 'crowdsourced' columns
        result_d2 = process_txt_file('file_path_d2.txt', ['raw', 'corrected', 'crowdsourced'])
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        sub_documents = content.split('*$*OVERPROOF*$*')[1:]  # Split the content into sub-documents
        result = []
        for sub_doc in sub_documents:
            lines = sub_doc.strip().split('\n')
            metadata = lines[0].strip()
            data = {column: [] for column in column_names}
            for line in lines[1:]:
                parts = line.split('||@@||')
                for i, part in enumerate(parts):
                    data[column_names[i]].append(part.strip())
                for column in column_names[len(parts):]:
                    data[column].append(parts[-1].strip())
            result.append({
                'metadata': metadata,
                **{column: '\n'.join(data[column]) for column in column_names}
            })
        return result