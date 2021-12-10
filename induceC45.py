import numpy as np
import pandas as pd
import math  # for log calculation
import json
import sys
import time
# Requires a path to a TRAINING set, which includes the class label in the dataset.
# Structure of our dataset CSV file should be as follows:
#   Line 1 -> Attributes of our dataset, including out class category
#   Line 2 -> Domain of each attribute
#   Line 3 -> Our class attribute
#   Line 4+ -> Our data
# Returns a pandas dataframe of the dataset, and a dictonary containing our attributes.
# This dictionary also contains a 'class_label' key for easy access for later.


def csv_to_df(path):
    df = pd.read_csv(path, header=0)  # , skiprows=[2])
    attributes = df.iloc[0].to_dict()
    attributes['class_label'] = df.iloc[1][0]
    df = df.drop([0, 1])
    # replace values of attributes dict with possible values
    for key in attributes.copy():
        if key == 'class_label':
            continue
        # check if value isn't numerical
        if int(attributes[key]) != 0:
            val = df[key].unique()
        else:
            val = "num"
            df[key] = pd.to_numeric(df[key], errors='coerce')
        attributes[key] = val
    return df, attributes

# Requires a pandas dataframe and a dictionary object containing our
# attributes.
# Returns an array containing:
#   0 -> The most frequent class label (str)
#   1 -> The probability of choosing this label within the dataset.
# This is a helper function used by c45() when our information gain
# computations are not high enough for our threshold.


def find_most_frequent_label(dataframe, attributes):
    values = dataframe[attributes['class_label']
                       ].value_counts().keys().tolist()
    counts = dataframe[attributes['class_label']
                       ].value_counts().tolist()
    return [values[0], counts[0]/len(dataframe)]

# Determines which attribute to split on. Requires a pandas
# dataframe, a dictionary of attributes, and a float value containing
# our information gain threshold.
# Returns the name of the attribute to split on, or None if
# our information gain calculation was insufficient based on the
# threshold.


def select_splitting_attribute(dataframe, attributes, threshold):
    attributes_gain = []
    class_count = dataframe.groupby(attributes['class_label']).size()
    df_entropy = sum([-(x / len(dataframe)) * math.log2(x / len(dataframe))
                      for x in class_count])
    for attribute in dataframe.columns:
        # skip our class label -> it's not an attribute
        if attribute == attributes['class_label']:
            continue
        elif attribute not in attributes:
            # weird base case I encountered
            continue
        # check if attribute is categorical
        if type(attributes[attribute]) is np.ndarray:
            # print("cat")
            # for grabbing subset size for later
            attr_df = dataframe.groupby([attribute]).groups
            # group by labels in an attribute by the class label
            attr_class_df = dataframe.groupby(
                [attribute, attributes['class_label']]).groups
            attr_entropy = {}
            attr_subset_size = {}
            # will contain the gain for the attribute to append to attributes_gain array
            attribute_gain = {attribute: None}
            for key in attr_class_df:
                # key -> index for grouped attributes
                if key[0] not in attr_subset_size:
                    attr_subset_size[key[0]] = len(
                        attr_df[key[0]].tolist())
                if key[0] not in attr_entropy:
                    attr_entropy[key[0]] = 0
                pr = len(attr_class_df[key]) / len(attr_df[key[0]])
                attr_entropy[key[0]] += (math.log2(pr) * (-pr))
            # attribute gain calculation
            attribute_gain[attribute] = sum(
                [(attr_subset_size[key] / sum(attr_subset_size.values())) *
                 attr_entropy[key] for key in attr_entropy])
            attributes_gain.append(
                {'attribute': attribute, 'gain': df_entropy - attribute_gain[attribute]})
        else:
            num_gain = find_best_split(
                dataframe, attributes, attribute, df_entropy)
            # print("num attribute: {} thresh: {}".format(
            # attribute, num_gain[1]))

            attributes_gain.append(
                {'attribute': attribute, 'gain': num_gain[1], 'split': num_gain[0]})
    attributes_gain = sorted(
        attributes_gain, key=lambda d: d['gain'], reverse=True)
    # print(attributes_gain)
    return attributes_gain[0] if (len(attributes_gain) > 0 and attributes_gain[0]['gain'] > threshold) else None


# Determines the information gain for a numerical attribute.
# Calculates the best value to split on and returns the
# gain and value to split on.


def find_best_split(dataframe, attributes, num_attribute, df_entropy):
    split_vals = dataframe.copy(deep=True)[num_attribute].drop_duplicates(
    ).sort_values(ascending=True)
    df_pairs = [(dataframe[dataframe[num_attribute] <= x],
                 dataframe[dataframe[num_attribute] > x], x, len(dataframe[dataframe[num_attribute] <= x]), len(dataframe[dataframe[num_attribute] > x])) for x in split_vals]
    entr_pairs = [(x[0][attributes['class_label']].value_counts().apply(lambda class_value: -(class_value/len(x[0][attributes['class_label']])) * math.log2(class_value/len(x[0][attributes['class_label']]))), x[1]
                   [attributes['class_label']].value_counts().apply(lambda class_value: -(class_value/len(x[1][attributes['class_label']])) * math.log2(class_value/len(x[1][attributes['class_label']]))), x[2], x[3], x[4]) for x in df_pairs]

    gains = [(z, df_entropy - ((a/len(dataframe)) * x.sum() + (b/len(dataframe)) * y.sum()))
             for x, y, z, a, b in entr_pairs]
    gains = sorted(gains, key=lambda pair: pair[1], reverse=True)
    return [gains[0][0], gains[0][1]]


# The heart of our classification.
# Returns a classification tree as a dictionary.
# Requires a pandas dataframe, dictionary containing our attributes,
# and a threshold value.
# The parent and file path arguments are to place metadata in the
# tree when it is initialized.


def c45(dataframe, attributes, threshold, parent=False, file=None):
    attr_copy = attributes.copy()
    node_label = None
    tree = {}
    if parent:
        # just to add our metadata for the tree
        tree['dataset'] = file
    # check termination conditions 1
    if len(dataframe.groupby(attributes['class_label'])) == 1:
        # print('leaf created: same class label')
        node_label = list(dataframe.groupby(
            [attributes['class_label']]).groups)[0]
        tree['leaf'] = {'decision': node_label, 'p': 1.0}
    # check termination condition 2
    elif len(attributes) == 1:
        # print('leaf created: no more attributes')
        node_label = find_most_frequent_label(dataframe, attributes)
        tree['leaf'] = {'decision': node_label[0], 'p': node_label[1]}
    else:
        # determine splitting attribute
        splitting_attribute = select_splitting_attribute(
            dataframe, attributes, threshold)
        # check if we don't have an attribute higher than our threshold
        if not splitting_attribute:
            # print('leaf created: not high enough gain')
            node_label = find_most_frequent_label(dataframe, attributes)
            tree['leaf'] = {'decision': node_label[0], 'p': node_label[1]}
        # split on the attribute
        else:
            node_label = splitting_attribute['attribute']
            # Split numerical category if applicable
            if 'split' in splitting_attribute:
                tree['node'] = {'var': node_label, 'edges': []}
                if type(node_label) == np.NaN:
                    print("ERROR: {}".format(node_label))
                df_l = dataframe[dataframe[node_label] <=
                                 float(splitting_attribute['split'])]
                df_r = dataframe[dataframe[node_label] >
                                 float(splitting_attribute['split'])]
                if len(df_l) == 0:
                    freq_label = find_most_frequent_label(
                        dataframe, attributes)
                    l_tree = {
                        'leaf': {'decision': freq_label[0], 'p': freq_label[1]}}
                else:
                    l_tree = c45(df_l, attr_copy, threshold)

                if len(df_r) == 0:
                    freq_label = find_most_frequent_label(
                        dataframe, attributes)
                    r_tree = {
                        'leaf': {'decision': freq_label[0], 'p': freq_label[1]}}
                else:
                    r_tree = c45(df_r, attr_copy, threshold)

                l_val = 'leaf' if 'leaf' in l_tree else 'node'
                tree['node']['edges'].append(
                    {'edge': {'value': splitting_attribute['split'], 'direction': 'le', l_val: l_tree[l_val]}})
                # r_tree = c45(df_r, attr_copy, threshold)
                r_val = 'leaf' if 'leaf' in r_tree else 'node'
                tree['node']['edges'].append(
                    {'edge': {'value': splitting_attribute['split'], 'direction': 'gt', r_val: r_tree[r_val]}})
            # Split categorical attribute if applicable
            else:
                attr_copy.pop(node_label, None)
                tree['node'] = {'var': node_label, 'edges': []}
                for key in attributes[node_label]:
                    filtered_df = dataframe.loc[dataframe[node_label] == key]
                    if len(filtered_df) > 0:
                        val = (c45(filtered_df, attr_copy, threshold))
                        result_label = 'node' if 'node' in val else 'leaf'
                        tree['node']['edges'].append(
                            {
                                'edge': {
                                    'value': key, result_label: val[result_label]}})
                    # handle ghost leaf
                    else:
                        freq_label = find_most_frequent_label(
                            dataframe, attributes)
                        tree['node']['edges'].append({
                            'edge': {'value': key, 'leaf': {'decision': freq_label[0], 'p': freq_label[1]}}})
    return tree

# Our wrapper to create a c45 decision tree. Requires a path to our dataset and
# an option restrictions file containing attributes to ignore during our classification.


def induce_c45(data_path, restrictions_path=None):
    df, attributes = csv_to_df(data_path)
    tree = c45(df, attributes, 0.25, True, data_path)
    print(json.dumps(tree, indent=2))
    return tree


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python3 induceC45.py <{}> [<{}>]".format(
            "dataset", "restrictions"))
    elif ".csv" not in args[0]:
        print("Please provide a csv file as the first argument.")
    else:
        induce_c45(args[0], args[1] if len(args) > 1 else None)
