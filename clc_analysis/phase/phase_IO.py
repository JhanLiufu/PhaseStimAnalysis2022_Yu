import pandas as pd
import pickle as pkl
import numpy as np
import os
from os.path import isdir


def read_day_param(fname_session_param, dates):
    """
    Read from an input Excel spreadsheet the CLC detection parameters on the specified dates

    Parameters
    ----------
    fname_session_param : string, full file directory of input parameter spreadsheet
    dates : list of strings, list of dates to include

    Returns
    -------
    pandas dataframe of CLC detection parameters in sessions of the specified dates
    """
    all_param = pd.read_excel(fname_session_param, engine='openpyxl')

    day_incl = []
    for ind, row in all_param.iterrows():
        if str(row['Date']) in dates:
            day_incl.append(ind)

    return all_param.iloc[day_incl]


def get_unanalyzed(fname_session_param, results_dir):
    """
    Check which session's CLC data has not been analyzed by checking existence of directory
    in results folder

    Parameters
    ----------
    fname_session_param : string, full file directory of input parameter spreadsheet
    results_dir : string, full directory of analysis results folder

    Returns
    -------
    pandas dataframe of CLC parameters in the unanalyzed sessions
    """
    all_param = pd.read_excel(fname_session_param, engine='openpyxl')

    session_incl = []
    for ind, row in all_param.iterrows():
        session = row['Session']
        session_dir = str(row['Date']) + f'_s{session}'
        if not isdir(results_dir+session_dir):
            session_incl.append(ind)

    return all_param.iloc[session_incl]


def load_pkl_in_session(dir_sim_session, f_name, keep_dir=True):
    if keep_dir:
        res = {}
    else:
        res = []
        
    for x in os.walk(dir_sim_session):
        try:
            curr_dir = f'{x[0]}'
            with open(f'{curr_dir}/{f_name}.pkl', 'rb') as curr_f:
                if keep_dir:
                    res[os.path.basename(curr_dir)] = pkl.load(curr_f)
                else:
                    res.append(pkl.load(curr_f))
        except FileNotFoundError:
            pass
        
    return res


def get_field_array(metrics_dict_lst, field_name):
    res = []
    for i in metrics_dict_lst:
        res.append(i.get(field_name))
        
    return np.array(res)


def group_dicts_by_subject(data_list, substring_index=1):
    grouped_data = {}

    for data_dict in data_list:
        for key, value in data_dict.items():
            substrings = key.split('_')
            
            if len(substrings) > substring_index:
                substring = substrings[substring_index]
                
                # Create a list for the substring if it doesn't exist
                if substring not in grouped_data:
                    grouped_data[substring] = []
                
                # Append the dictionary to the corresponding list
                grouped_data[substring].append({key: value})
    
    return grouped_data


def average_nested_dicts(lst_of_dicts, ignore_fields=None):
    # Initialize an empty dictionary to store the averaged values
    averaged_dict = {}

    # If ignore_fields is not provided, initialize it as an empty list
    ignore_fields = ignore_fields or []

    for outer_dict in lst_of_dicts:
        for _, inner_dict in outer_dict.items():
            # print(inner_dict)
            # Iterate over the keys (inner dictionary keys) in each inner dictionary
            for key, value in inner_dict.items():
                # Check if the key is in the ignore_fields list
                if key in ignore_fields:
                    continue  # Skip this field if it's in the ignore list

                    # Special handling for 'avg_coherence'
                if key == 'avg_cohernce':
                    if key in averaged_dict:
                        averaged_dict[key][0] += value[0]
                        averaged_dict[key][1] += value[1]
                    else:
                        averaged_dict[key] = [value[0], value[1]]
                else:
                    # print('here')
                    # Check if the key is already in the averaged_dict
                    if key in averaged_dict:
                        # If the value is a numerical type, add it to the running sum
                        if isinstance(value, (int, float)):
                            # print('here')
                            averaged_dict[key] += value
                        # If the value is a tuple (like 'avg_coherence'), element-wise add
                        elif isinstance(value, tuple):
                            averaged_dict[key] = tuple(x + y for x, y in zip(averaged_dict[key], value))
                    else:
                        # If the key is not in averaged_dict, add it with the current value
                        averaged_dict[key] = value
                        
    # Calculate the average for numerical values
    for key, value in averaged_dict.items():
        if key == 'avg_coherence':
            averaged_dict[key] = (value[0] / len(lst_of_dicts), value[1] / len(lst_of_dicts))
        elif isinstance(value, (int, float)):
            averaged_dict[key] /= len(lst_of_dicts)
        else:
            # Handle other types if needed
            pass
                        
    return averaged_dict
                        

def read_from_session(dir_ses, fname, ignore_fields=None, substring_index=1):
    dicts_by_subj = group_dicts_by_subject([load_pkl_in_session(dir_ses, fname)], substring_index=substring_index)
    lst_dicts_subj_avg = [average_nested_dicts(v, ignore_fields=ignore_fields) for _, v in dicts_by_subj.items()]
    return lst_dicts_subj_avg

