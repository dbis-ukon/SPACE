import numpy as np
import torch
from unidecode import unidecode
import pandas as pd


def min_max_scale_overall(file_to_normalize,train_path, test_path, feature_range=(0, 1) ):
    card_list = []
    line_list = []
    for line in open(file_to_normalize):
        line = line.strip().split(':')
        k = list(map(int, line[-1].split(' ')))  # Convert strings to integers
        card_list.append(k)  # Append to the main list
        line_list.append(line[0].replace(' ', '/'))

    # Flatten the list of lists to get all values in a single list
    flat_list = [item for sublist in card_list for item in sublist]

    # Find the global minimum and maximum values across all vectors
    global_min = min(flat_list)
    global_max = max(flat_list)

    # Apply log transformation to flatten the skewness
    log_flat_list = [np.log1p(x) for x in flat_list]  # log1p is log(1 + x)

    # Find the new minimum and maximum values after log transformation
    log_global_min = min(log_flat_list)
    log_global_max = max(log_flat_list)

    # Perform Min-Max scaling on the log-transformed values
    a, b = feature_range
    def scale_value(x):
        if log_global_max > log_global_min:
            return float(a + (np.log1p(x) - log_global_min) * (b - a) / (log_global_max - log_global_min))
        else:
            return 0
    scaled_vectors = [[scale_value(x) for x in sublist] for sublist in card_list]
    full_paths= []
    for i in range(len(scaled_vectors)):
        formatted_vector = [f"{num:.20f}" for num in scaled_vectors[i]]
        full_paths.append(line_list[i] + ':' + ' '.join(formatted_vector))
    split_train_test(full_paths, train_path, test_path)
    return global_min, global_max


def min_max_descale(scaled_value, original_global_min, original_global_max, feature_range=(0, 1)):
    # Extract the feature range
    a, b = feature_range

    # Calculate the log of the original global min and max values (before scaling)
    log_global_min = np.log1p(original_global_min)  # log(1 + min)
    log_global_max = np.log1p(original_global_max)  # log(1 + max)

    # Function to reverse the scaling for a single value
    def inverse_scale_value(scaled_value):
        if log_global_max > log_global_min:
            # Reverse the min-max scaling
            log_value = log_global_min + (scaled_value - a) * (log_global_max - log_global_min) / (b - a)
            # Reverse the log transformation
            return np.expm1(log_value)  # expm1 is exp(x) - 1
        else:
            return 0  # Handle cases where max == min
    descaled_value = inverse_scale_value(scaled_value)

    return descaled_value


def split_train_test(data_list, train_path, test_path):
    # Shuffle the data
    np.random.shuffle(data_list)

    # Calculate the split index
    split_index = int(len(data_list) * 0.9)

    # Split the data
    train_data = data_list[:split_index]
    test_data = data_list[split_index:]

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    # Save to CSV
    train_df.to_csv(train_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)

#This function returns geometric mean of a list
def g_mean(list_):
    log_list_ = np.log(list_)
    return np.exp(log_list_.mean())


#This function computes loss
def binary_crossentropy(preds, paths_cardinalities, mask):
    loss = paths_cardinalities * torch.log(preds + 0.000000001) + (1 - paths_cardinalities) * torch.log(1 - (preds - 0.000000001))
    if mask is not None:
        loss = mask * loss
    return - torch.sum(loss) / torch.sum(mask)


def name_to_tensor(name, char2idx):
    name =name.split('/')
    tensor = torch.zeros(len(name), len(char2idx))
    for i, char in enumerate(name):
        tensor[i][char2idx[char]] = 1
    return tensor


#This function loads LIKE-patterns with ground truth probabilities
def load_train_data(filename, char2idx):
    input_paths = []
    paths_cardinalities = []
    path_length = []
    for line in open(filename):
        line_ = line.strip().split(':')
        transformed_to_tensor = name_to_tensor(unidecode(line_[0]), char2idx)
        input_paths.append(transformed_to_tensor)
        path_length.append(len(transformed_to_tensor))
        ground_prob_list = [float(element) for element in line_[-1].split(' ')]
        paths_cardinalities.append(ground_prob_list)
    return input_paths, paths_cardinalities, max(path_length)

#This function pads the zero vectors to like-patterns.
def add_pad_train_data(filename, char2idx):
    zeros_vector = [[0] * len(char2idx)]
    padded_input_paths = []
    input_paths, paths_cardinalities, maxs = load_train_data(filename,char2idx)
    for i in input_paths:
        old_len = len(i)
        for k in range(maxs - len(i)):
            i = torch.cat((i, torch.tensor(zeros_vector)), 0)
        padded_input_paths.append((i, [1] * old_len + [0] * (maxs - old_len)))
    paths_cardinalities1 = []
    for i in paths_cardinalities:
        paths_cardinalities1.append(i + (maxs - len(i)) * [0])
    train_dataset = [(torch.tensor(padded_input_paths[i][0]), torch.tensor(padded_input_paths[i][1]), torch.tensor(paths_cardinalities1[i])) for i in
                     range(len(paths_cardinalities))]
    return train_dataset


#This function takes a file path that contains test LIKE-patterns
def loadtestData(filename, char2idx):
    input_paths = []
    path_length = []
    actual_card = []
    with open(filename, 'r') as file:
        for line in file:
            line_ = line.strip().split(':')
            if len(line_[0].split('/')) > 0:
                actual_card.append([float(element) for element in line_[-1].split(' ')])
                transformed_to_tensor = name_to_tensor(unidecode(line_[0]), char2idx)
                input_paths.append(transformed_to_tensor)
                path_length.append(len(transformed_to_tensor))
    return input_paths, max(path_length), actual_card



def add_pad_test_data(filename, char2idx):
    liste = [[0] * len(char2idx)]
    padded_input_paths = []
    masks = []
    input_paths, maxs, actual_card = loadtestData(filename, char2idx)
    for i in input_paths:
        old_len = len(i)
        for k in range(maxs - len(i)):
            i = torch.cat((i, torch.tensor(liste)), 0)
        padded_input_paths.append(i)
        masks.append([1] * old_len + [0] * (maxs - old_len))
    test_dataset = [(torch.tensor(padded_input_paths[i]), torch.tensor(masks[i]), torch.tensor(actual_card[i])) for i in range(len(masks))]
    return test_dataset


##This function compute and return q-error
def compute_qerrors(actual_card, estimated_card):
    return max(actual_card/estimated_card, estimated_card/actual_card)


