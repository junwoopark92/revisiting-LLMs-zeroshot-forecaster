import numpy as np
from scipy import stats
import pickle


def cal_metric(pred, true):
    assert pred.shape == true.shape
    pred = np.array(pred)
    true = np.array(true)
    mse = np.mean(((pred - true) ** 2))
    mae = np.mean(np.abs(pred - true))
    return mse, mae

# Function to save object as pickle
def save_pickle(obj, filepath):
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object successfully saved to {filepath}.")
    except Exception as e:
        print(f"Error saving pickle: {e}")

# Function to load object from pickle file
def load_pickle(filepath):
    try:
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {filepath}.")
        return obj
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None


def calculate_mean_confidence_interval(data, confidence_level=0.95):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    n = len(data)

    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, df=n-1)

    margin_of_error = t_critical * (std_dev / np.sqrt(n))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    return mean, margin_of_error
