
import collections
import matplotlib
matplotlib.use("Qt4agg")
import matplotlib.pyplot as plt
import numpy
from sklearn.tree import DecisionTreeClassifier



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


# ['1', '-1', '-1', '-1', '1', '1']
training_data = process_data(read_data('data/arcene_train.data'))
training_labels = process_data(read_data('labels/arcene_train.labels'))


# ['1', '-1', '-1', '-1', '1', '1']
valid_labels = process_data(read_data('labels/arcene_valid.labels'))
valid_data = process_data(read_data('data/arcene_valid.data'))

training_errors = []
valid_errors = []
depth_list = []


for depth in range(1, 13):
    training_tree = DecisionTreeClassifier(max_depth=depth)
    training_tree = training_tree.fit(training_data, training_labels)

    """
    Create a decision tree and fit it with training data and corresponding
    training labels.

    Use that decision tree to guess/predict what labels the test_data will
    produce

    See if the labels that were predicted match the actual labels
    """
    valid_data_score = training_tree.score(valid_data, valid_labels)
    valid_data_error = round(1 - valid_data_score, 3)
    valid_errors.append(float("%.4f" % valid_data_error))

    training_data_score = training_tree.score(training_data, training_labels)
    training_data_error = round(1 - training_data_score, 3)
    training_errors.append(float("%.4f" % training_data_error))

    depth_list.append(depth)

    print("******************************")
    print("The valid data score is {}.".format(valid_data_score))
    print("The valid data error is {}.".format(valid_data_error))
    print("The training data score is {}.".format(training_data_score))
    print("The training data error is {}.".format(training_data_error))
    print("******************************")


print(training_errors)
print(valid_errors)
print(depth)

plt.plot(depth_list, training_errors)
plt.plot(depth_list, valid_errors)
plt.title('Depth vs Training and Valid Errors')
plt.xlabel('Depth')
plt.ylabel('Rate of Error: TE =Blue, VE=Green')
plt.plot
plt.show()
