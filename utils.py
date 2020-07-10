import csv
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

def get_file_size(file_path):
	"""Get Integer value for size of file in Bytes.

	Args:
		file_path ([str]): File name or full path

	Returns:
		[int]: Size of input file in Bytes

	Reference:
	https://stackoverflow.com/questions/6591931
	os.path.getsize(file_path)
	"""
	num_bytes = Path(file_path).stat().st_size
	return num_bytes


def list_all_files(directory_pattern):
	"""Get list of filenames for a given path.

	Args:
		directory_pattern ([str]): path as string

	Return:
		([list]) python list with filenames as elements
	"""
	file_paths = glob.glob(directory_pattern)
	return file_paths


def file_name_from_path(file):
	"""Read filepath string, remove directories and extensions.

	Args:
		file ([str]): file path to trim

	Returns:
		[str]: isolated filename from path
	"""
	noExtension = file.split('.')[0]
	fileName = noExtension.split('\\')[-1]
	return fileName


def list_filename_strings(paths):
	"""Return a list of filename from a list of filepaths.

	Args:
		paths ([list]): python list containing filepaths as strings

	Calls:
		file_name_from_path()

	Returns:
		[list]: python list of cleaned filenames
	"""
	file_names = list(map(lambda x: file_name_from_path(x), paths))
	return file_names


def handle_encoding_errors(filepath):
    """Handle special characters, byte strings.

    See:
    https://stackoverflow.com/questions/42339876/
    """
    with open(filepath, "rb") as f:
        contents = f.read()
    with open(filepath, encoding="utf8", errors="ignore") as f:
        contents = f.read()


def zero_dataframe(list_of_file_names, list_of_column_names):
	"""Generate a Dataframe with Filenames as Index,
	Columns as all Column Names.

	Args:
		list_of_file_names ([type]): [description]
		list_of_column_names ([type]): [description]

	Returns:
		[DataFrame]: df as matrix of zeroes
	"""
	zero_matrix = np.zeros((
		len(list_of_file_names),
		len(list_of_column_names)),
		dtype=int)

	zero_df = pd.DataFrame(
		zero_matrix,
		index=list_of_file_names,
		columns=list_of_column_names)

	return zero_df


def dict_files_cols_from_files(file_paths):
	"""Dict where keys are filenames, values are lists of column names.

	Args:
		file_paths ([list]): list of filepaths

	Returns:
		[dict]: dictionary with filenames and column names, lowercase
	"""
	file_summary_dict = {}
	for file_path in file_paths:
		file_name = file_name_from_path(file_path)
		df = pd.read_csv(file_path, header=0, encoding='utf-8')
		cols = [col.lower() for col in df.columns]
		file_summary_dict[file_name] = cols

	return file_summary_dict


def dict_as_json(dict):
	"""Format dictionary as newline delimited JSON.

	Args:
		dict ([dict]): Any python dictionary

	Returns:
		[type]: JSON string, newline delimited
	"""
	json_output = json.dumps(dict, sort_keys=True, indent=4)

	return json_output


def uniq_column_names(dict):
	"""Finds all unique values found in lists in python dict.

	Args:
		dict ([dict]): dictionary expected to have lists as values

	Returns:
		[list]: list of all unique values found in dict
	"""
	all_col_names = []
	for file in dict.keys():
		cols = dict[file]
		all_col_names = all_col_names + cols
	uniq_col_names = set(all_col_names)

	return uniq_col_names


def dateframe_of_file_columns(file_dict):
	# accepts a dictionary
	# where keys are filenames (string)
	# value is list of column names (strings)
	file_columns_df = pd.DataFrame.from_dict(file_dict, orient='index')

	return file_columns_df


def output_file_col_matrix(file_object):
	# accepts a dictionary
	# where keys are filenames (string)
	# value is list of column names (strings)
	file_names = file_object.keys()
	all_col_names = uniq_column_names(file_object)

	matrix = zero_dataframe(file_names, all_col_names)

	for key in file_object:
		for val in file_object[key]:
			matrix[val][key] = 1

	return matrix


def csv_to_df(file):
    """Reading CSV file to established DataFrame.

    """
    # Index column as PID
    # na_values = ['value_to_replace1', 'value_to_replace2']
    df = pd.read_csv(file,
                    header=0,
                    index_col=0,
                    encoding='utf-8'
                    )
    return df


def df_to_csv(df, columns, filename):
    """Write df to CSV file with filename and desired columns in list.

    Args:
        df: pandas dataframe
        columns: columns to include, as list of strings
        filename: desired name of file as string
    """
    output_df = df[:,[columns]]

    output_df.to_csv(filename,
            header=True,
            encoding='utf-8')


def replace_unichar_in_column(df, col_name, to_replace, replace_with):
    """Replaces problem unicode characters in a whole column.

    Args:
        df: input pandas dataframe
        col_name: column to search through
        to_replace: character to replace, if unicode use u""
        replace_with: desired replacement character

    Returns:
        New dataframe with the special character removed / replaced
    """
    df[col_name] = df[col_name].str.replace(to_replace, replace_with)
    return df


def max_num_cols_in_rows(csv_file):
    """Gather Maximum number of Columns present in CSV.


    """

def csv_summary_stats(file_path):
    """Read in CSV, output summary stats on columns and values.

    Args:
        file_path ([str]): Path for desired file for analysis

    Read CSV into DataFrame, perform summary statistics on each column.
	Output results as file.
	Numeric Columns: min, max, num of blanks/zeros,
	Discrete: unique values, num of unique values, min/max of length of strings
	https://pandas.pydata.org/docs/getting_started/intro_tutorials/06_calculate_statistics.html
    """
    df = pd.read_csv(file_path, header=0, encoding='utf-8')

    report_df = df.describe(include='all')
    filename = file_name_from_path(file_path)

    report_df.to_csv(
		filename + '_pd_describe.csv',
		header=True,
		encoding='utf-8'
		)


def df_summary_stats(df):
    """Read in CSV, output summary stats on columns and values.

    Args:
        file_path ([df]): pandas DataFrame to summarize

    Return:
        Writes CSV of results
        
    Read CSV into DataFrame, perform summary statistics on each column.
	Output results as file.
	Numeric Columns: min, max, num of blanks/zeros,
	Discrete: unique values, num of unique values, min/max of length of strings
	https://pandas.pydata.org/docs/getting_started/intro_tutorials/06_calculate_statistics.html
    """
    report_df = df.describe(include='all')
    filename = file_name_from_path(file_path)

    report_df.to_csv(
		filename + '_pd_describe.csv',
		header=True,
		encoding='utf-8'
		)

