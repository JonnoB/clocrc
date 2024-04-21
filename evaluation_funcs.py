import jellyfish
import numpy as np
import pandas as pd
import os

def vectorized_levenshtein_similarity(leven_distances, lengths):
    """
    Vectorized calculation of Levenshtein similarity based on arrays of Levenshtein distances
    and the corresponding lengths of strings.

    Parameters:
    - leven_distances (np.array): An array of Levenshtein distances.
    - lengths (np.array): An array of the maximum lengths between compared strings.

    Returns:
    - np.array: An array of Levenshtein similarities.
    """
    # Ensure zero division is handled by returning 0 similarity in such cases
    similarities = np.where(lengths > 0, (lengths - leven_distances) / lengths, 0)
    return similarities

# Example usage within the calculate_metrics function
def calculate_metrics(file_name, raw_data_string, contents, wer, cer):
    wer_score = wer.compute(predictions=[raw_data_string], references=[contents])
    cer_score = cer.compute(predictions=[raw_data_string], references=[contents])
    leven_score = jellyfish.levenshtein_distance(raw_data_string, contents)
    
    # Here we prepare for vectorized computation
    leven_distances = np.array([leven_score])
    lengths = np.array([max(len(raw_data_string), len(contents))])
    
    #This is not very useful as I can use the error rate reduction
    #lev_sim = vectorized_levenshtein_similarity(leven_distances, lengths)[0]  # Get the first element since we're dealing with single values

    results_df = pd.DataFrame({
        'File Name': [file_name],
        'WER': [wer_score],
        'CER': [cer_score],
        'lev_dist': [leven_score]#,
    #    'lev_sim': [lev_sim]
    })

    return results_df


def evaluate_ocr_dataframe(dev_data_raw_df, dev_transcripts, wer, cer):
    """
    Evaluate a dataframe to calculate Word Error Rate (WER), Character Error Rate (CER),
    and Levenshtein Distance for each entry. It assumes each entry includes a file name
    pointing to a transcript file, and raw data content for comparison.

    WARNING: Only pre-processing is convert to lower, on bot raw data and recovered data

    Parameters:
    - dev_data_raw_df (pd.DataFrame): A DataFrame containing the dataset to be processed.
      It must include 'file_name' and 'content_html' columns.
    - dev_transcripts (str): The directory path where the transcript files referenced in
      dev_data_raw_df are stored.

    Returns:
    - pd.DataFrame: A combined DataFrame containing the file name, WER, CER, and Levenshtein
      Distance for each file in the dataset.

    This function iterates through each row in the input DataFrame, reads the transcript file
    contents, and uses the `calculate_metrics` function to calculate the necessary metrics. It
    compiles the results from all files into a single DataFrame.
    """

    results_list = []  # To store the dataframes generated for each file
    
    # Loop through each row in the DataFrame
    for index, row in dev_data_raw_df.iterrows():
        file_name = row['file_name']
        raw_data_string = row['content_html']
        file_path = os.path.join(dev_transcripts, file_name)

        # Open the file and read its contents
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.read()

        # Calculate metrics and append the resulting dataframe to the list
        metrics_df = calculate_metrics(file_name, raw_data_string.lower(), contents.lower(), wer, cer)
        results_list.append(metrics_df)
    
    # Combine all the DataFrames in the list into a single DataFrame
    combined_df = pd.concat(results_list, ignore_index=True)
    
    return combined_df


def load_txt_files_to_df(directory):
    """
    Loads the content of all '.txt' files within a specified directory into a pandas DataFrame.

    Parameters:
    - directory (str): The path to the directory containing '.txt' files.

    Returns:
    - pd.DataFrame: A DataFrame with a single column "content_html", where each row contains
      the content of one '.txt' file from the directory.
    """
    # Initialize a list to store the content of each text file
    content_list = []
    
    # Loop through each file in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a '.txt' file
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            # Open the file and read its contents
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                content_list.append({'content_html': content, 'file_name':file_name})
    
    # Create a DataFrame with the contents
    df = pd.DataFrame(content_list)
    
    return df

def percentage_error_reduction(old_errors, new_errors):
  """
  Calculates the percentage reduction in error between two vectors.

  Args:
      old_errors (numpy.ndarray): A numpy array containing the old error rates.
      new_errors (numpy.ndarray): A numpy array containing the new error rates.

  Returns:
      numpy.ndarray: A numpy array containing the percentage reduction for each element.
  """

  # Handle division by zero with np.where
  zero_mask = np.where(old_errors == 0, True, False)
  reduction = np.where(zero_mask, float('inf'), (old_errors - new_errors) / old_errors * 100)
  return reduction


def get_metric_error_reduction(corrected_df, raw_df):
    """
    Calculate the percentage error reduction for WER, CER, and lev_dist between corrected and raw DataFrames.

    Args:
        corrected_df (pandas.DataFrame): DataFrame containing the corrected metrics (WER, CER, lev_dist).
        raw_df (pandas.DataFrame): DataFrame containing the raw metrics (WER, CER, lev_dist).

    Returns:
        pandas.DataFrame: A new DataFrame with columns 'File Name', 'WER_reduction', 'CER_reduction',
                          and 'lev_dist_reduction', representing the percentage error reduction for each file
                          and metric.

    Example:
        >>> corrected_df = pd.DataFrame({'File Name': ['file1', 'file2'], 'WER': [0.1, 0.2], 'CER': [0.05, 0.1], 'lev_dist': [10, 20]})
        >>> raw_df = pd.DataFrame({'File Name': ['file1', 'file2'], 'WER': [0.2, 0.3], 'CER': [0.1, 0.15], 'lev_dist': [20, 30], 'type': ['raw', 'raw']})
        >>> get_metric_error_reduction(corrected_df, raw_df)
           File Name  WER_reduction  CER_reduction  lev_dist_reduction
        0      file1           50.0           50.0                50.0
        1      file2           33.3           33.3                33.3
    """
    merged_df = corrected_df.merge(raw_df.drop(columns='type'), on='File Name', suffixes=['_corrected', '_raw'])
    
    # Calculate the percentage error reduction for each column
    wer_reduction = percentage_error_reduction(merged_df['WER_raw'], merged_df['WER_corrected'])
    cer_reduction = percentage_error_reduction(merged_df['CER_raw'], merged_df['CER_corrected'])
    lev_dist_reduction = percentage_error_reduction(merged_df['lev_dist_raw'], merged_df['lev_dist_corrected'])
    
    # Create a new DataFrame with the calculated relative errors
    relative_errors_df = pd.DataFrame({
        'File Name': merged_df['File Name'],
        'type':merged_df['type'],
        'WER': wer_reduction,
        'CER': cer_reduction,
        'lev_dist': lev_dist_reduction
    })
    
    return relative_errors_df


def evaluate_correction_performance(folder, transcripts_dir, wer_func, cer_func, type):
    """
    Evaluate the performance of the OCR relative to the transcript ground truth.

    Args:
        folder (str): The directory path where the raw OCR text files are located.
        transcripts_dir (str): The directory path where the corresponding transcripts are located.
        wer_func (function): The function to calculate the Word Error Rate (WER).
        cer_func (function): The function to calculate the Character Error Rate (CER).
        type (str): The type of data being processed.

    Returns:
        pandas.DataFrame: The processed DataFrame containing the evaluation metrics.

    """
    dev_data_raw_df = load_txt_files_to_df(folder)
    
    # filter to only have the ones with a transcript
    dev_data_raw_df = dev_data_raw_df.loc[dev_data_raw_df['file_name'].isin(os.listdir(transcripts_dir))].reset_index(drop=True)
    
    eval_temp = evaluate_ocr_dataframe(dev_data_raw_df, transcripts_dir, wer_func, cer_func)
    eval_temp['type'] = type
    
    return eval_temp

def evaluate_correction_performance_folders(corrected_folder, transcript_folder, wer_func, cer_func):
    """
    Calls `evaluate_correction_performance` on a folder of folders, allows the comparison of multiple models or prompts
    Evaluate the performance of LLM post-OCR recovery using WER, CER, and levenstiend distance

    This function uses the files in the corrected and transcript folders to evaluate how well the post-OCR correction has gone.
    Each file is evaluated and tagged with the file name for easy reference, sub-folders are tagged as 'type'

    Args:
        corrected_folder (str): The folder containing the corrected data.
        transcript_folder (str): The folder containing the ground truth transcripts.
        wer_func (function): The function to calculate the Word Error Rate (WER).
        cer_func (function): The function to calculate the Character Error Rate (CER).

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each folder.

    """
    performance_eval = []
    for folder in os.listdir(corrected_folder):
        
        eval_temp = evaluate_correction_performance(os.path.join(corrected_folder, folder),transcript_folder , wer_func, cer_func, folder)

        performance_eval.append(eval_temp)

    performance_eval = pd.concat(performance_eval)
    return performance_eval