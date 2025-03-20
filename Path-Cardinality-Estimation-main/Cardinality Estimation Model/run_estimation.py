from collections import namedtuple
import torch
from torch.utils.data import DataLoader
import utils
from cardinality_estimator import Path_Cardinality_Estimator
from model import train_model
import numpy as np
import os


def set_vocabulary(file_to_train):
    """Retrieve unique vocabulary from a path pattern file.

    Args:
        file_to_train (str): Path to the path patterns.

    Returns:
        str: A string representation of the unique vocabulary.
    """
    unique_words = set()  # Use a set to automatically handle duplicates
    with open(file_to_train, 'r') as file:
        for line in file:
            words = line.strip().split(':')[0].split(' ')
            unique_words.update(words)

    vocabulary = '/'.join(unique_words)
    return vocabulary


def train_estimator(train_data, model, device, learning_rate, num_epochs, model_save_path):
    """
    Train the estimation model and save it.

    Args:
        train_data (torch.utils.data.Dataset): Training dataset.
        model (torch.nn.Module): Estimation model to be trained.
        device (torch.device): Device for computation.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train.
        model_save_path (str): Path to save the trained model.

    Returns:
        torch.nn.Module: The trained estimation model.
    """
    model = train_model(train_data, model, device, learning_rate, num_epochs)
    torch.save(model.state_dict(), model_save_path)
    return model


def load_estimation_model(model_file_name, model, device):
    """
    Load a saved estimation model.

    Args:
        model_file_name (str): Path to the saved model file.
        model (torch.nn.Module): Estimation model to load parameters into.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: The loaded estimation model on the specified device.
    """
    if os.path.exists(model_file_name):
        model.load_state_dict(torch.load(model_file_name))
        return model.to(device)
    else:
        raise FileNotFoundError(f"Model file not found: {model_file_name}")



def estimate_path_cardinality(test_dataset, model, device, minimum, maximum):
    """

    Estimate cardinality using the trained model and print qerrors.

    Args:
        test_dataset (torch.utils.data.Dataset): Dataset for evaluation.
        model (torch.nn.Module): Trained estimation model.
        device (torch.device): Device for computation.
        minimum (float): Minimum value for scaling.
        maximum (float): Maximum value for scaling.
    """
    with torch.no_grad():
        path_qerror_list = []
        for name, pad_mask, actual_card in test_dataset:
            name, pad_mask, actual_card = name.to(device), pad_mask.to(device), actual_card.to(device)
            result = torch.pow(model(name), pad_mask)
            estimate = utils.min_max_descale(float(result[0][len(actual_card[0])-1].item()),minimum, maximum)
            actual = utils.min_max_descale(float(actual_card[0][-1].item()),minimum, maximum)
            path_qerror_list.append(utils.compute_qerrors(actual, estimate))
        print(f'g-mean: {np.round(utils.g_mean(path_qerror_list), 4)}')
        print(f'Mean: {np.round(np.average(path_qerror_list), 4)}')
        print(f'Median: {np.round(np.percentile(path_qerror_list, 50), 4)}')
        print(f'90th: {np.round(np.percentile(path_qerror_list, 90), 4)}')
        print(f'99th: {np.round(np.percentile(path_qerror_list, 99), 4)}')



def main():
    path_patters_path = 'path to path patterns'
    train_path = 'path to csv file to save train data'
    test_path= 'path to csv file to save test data'
    save_model_path = 'path to model file to save trained model'
    vocabulary = set_vocabulary(path_patters_path)
    minimum, maximum = utils.min_max_scale_overall(path_patters_path, train_path, test_path)

    path_configs = namedtuple('path_configs', ['vocabulary', 'hidden_size', 'learning_rate', 'batch_size',
                                                 'num_epochs', 'train_data_path', 'test_data_path',
                                                 'device', 'save_path'])

    path_card_estimator_configs = path_configs( vocabulary=vocabulary,hidden_size=256,learning_rate=0.0001,
                                                batch_size=128,
                                                num_epochs=256,
                                                train_data_path=train_path,
                                                test_data_path=test_path,
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                save_path=save_model_path)

    label_dictionary = {label: i for i, label in enumerate(path_card_estimator_configs.vocabulary.split('/'))}
    model = Path_Cardinality_Estimator(1, path_card_estimator_configs.hidden_size,
                                  path_card_estimator_configs.device, len(label_dictionary))
    train_data = utils.add_pad_train_data(path_card_estimator_configs.train_data_path, label_dictionary)
    dataloadertrain = DataLoader(train_data, batch_size=path_card_estimator_configs.batch_size, shuffle=True)
    trained_model = train_estimator(dataloadertrain, model, path_card_estimator_configs.device,
                               path_card_estimator_configs.learning_rate, path_card_estimator_configs.num_epochs,
                               path_card_estimator_configs.save_path)


    test_dataset = utils.add_pad_test_data(path_card_estimator_configs.test_data_path,
                                     label_dictionary)
    dataloader_test = DataLoader(test_dataset, batch_size=1)
    est = estimate_path_cardinality(dataloader_test, trained_model, path_card_estimator_configs.device, minimum, maximum)
    return est


if __name__ == "__main__":
    main()
