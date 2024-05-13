import ast
import os
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

def generate_float_series(start: float, end: float, step: float) -> list[float]:
    """
    Generates a list of floats including a `start` and `end` point, and intermediate points `step` apart.

    :param start: Starting point.
    :param end: Ending point.
    :param step: Interval to generate series of floats between `start` and `end`.
    :return: List of floats sampling from the interval between `start` and `end`, each `step` apart.
    """
    # Ensure step is a positive float
    step = abs(step)

    # Initialize the series with the start value
    series = [start]

    # Generate numbers in the series
    while start + step <= end:
        start += step
        series.append(start)

    # Check if the end value is already in the series
    if series[-1] != end:
        series.append(end)

    return series

def convert_to_list(string):
    try:
        return ast.literal_eval(string)
    except:
        return []  # Return an empty list in case of error

def read_large_csv(file_path, columns_str2list=[], nrows=None, chunk_size=100, ):
    """
    Read a large CSV file in chunks.
    
    Args:
    """
    
    # Define the file path
    # file_path = f'../dataset/egoclip_groups_groupby_{grouping_type}.csv'

    # Determine the number of rows in the CSV
    total_rows = sum(1 for _ in open(file_path, 'r'))

    # Read the CSV with a progress bar
    # chunk_size = 100  # Adjust chunk size based on your needs
    chunks = []
    converters = None
    if columns_str2list != []:
        converters = {cname: convert_to_list for cname in columns_str2list}
    # Use tqdm to show progress
    for chunk in tqdm(pd.read_csv(file_path, 
                                chunksize=chunk_size,
                                index_col=0,
                                converters=converters,
                                nrows=nrows), 
                    total=total_rows/chunk_size,
                    unit=f'chunk ({chunk_size} videos per chunk)'):
        chunks.append(chunk)

    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunks, axis=0)

    # Now you can use df as a normal DataFrame
    return df    

def list_files_by_extension(directory, extension):
    """
    List all files of given extension in the given directory.

    :param directory: The path to the directory to search for files.
    :param extension: The extension to check for, e.g., ".json".

    :return: A list of paths to JSON files in the specified directory.
    """
    # List to hold the names of JSON files
    json_files = []

    # Walk through the directory
    for filename in os.listdir(directory):
        if filename.endswith(extension):  # Check for files ending with .json
            json_files.append(os.path.join(directory, filename))  # Append the full path of the file

    return json_files

def count_subdirectories(dir: str) -> int:
    if os.path.exists(dir):
        p = Path(dir)
        return len([x for x in p.iterdir() if x.is_dir()])
    else:
        return 0
    
def split_list_into_partitions(lst, n):
    """
    Split a list into n partitions as evenly as possible.

    :param lst: The list to split.
    :param n: The number of partitions to create.
    :return: A list of partitions, each a list of elements from the original list.
    """
    # Ensure that we do not exceed the number of elements in the original list
    n = min(n, len(lst))

    # Calculate the size of each partition
    base_size = len(lst) // n
    remainder = len(lst) % n

    # Initialize variables for partitioning
    partitions = []
    start_index = 0

    for i in range(n):
        # Determine the size of the current partition
        current_partition_size = base_size + (1 if i < remainder else 0)
        
        # Append the current partition to the partitions list
        partitions.append(lst[start_index:start_index + current_partition_size])
        
        # Move the start index to the next chunk of the list
        start_index += current_partition_size

    return partitions

def copy_directory_contents(source, target):
    """
    Copies contents of source directory into target directory.

    :param source: Source directory.
    :param target: Target directory.
    """

    # Ensure the target directory exists
    if not os.path.exists(target):
        os.makedirs(target)
        print(f"Created target directory {target}.")

    # List each item in the source directory
    for item in os.listdir(source):
        source_item = os.path.join(source, item)
        target_item = os.path.join(target, item)

        # Check if the item is a directory
        if os.path.isdir(source_item):
            # Recursively copy the entire directory
            shutil.copytree(source_item, target_item)
        else:
            # Copy the file
            shutil.copy2(source_item, target_item)

        print(f"Copied {source_item} to {target_item}.")