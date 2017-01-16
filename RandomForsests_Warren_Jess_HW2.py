import collections
import matplotlib
matplotlib.use("Qt4agg")
import matplotlib.pyplot as plt
import nummpy
from sklearn.tree import RandomForestClassifier

def read_data(filename):
    with open(filename,'r') as f:
        file_data = f.read().splitlines()
        # automatically closes the file here
    return file_data


def process_data(data_list):
    """ data_list is a list containing the list of strings of numbers.

    """
    # [['128', '19', '0', '192',...], ['128', '19', '0', '192',...],...]
    new_data = [group.split() for group in data_list]
    # type convert all the strings in each list in data into an integer
    for i, group in enumerate(new_data):
        new_data[i] = list(map(int, group))

    return new_data
