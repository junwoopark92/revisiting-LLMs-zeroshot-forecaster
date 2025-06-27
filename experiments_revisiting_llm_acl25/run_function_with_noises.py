import os
import sys
import time
import yaml
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import argparse

from ts_argparse import informer_parser


# Initial parsing of arguments to get chosen_model and chosen_dataset
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='experiments_revisiting_llm_acl25/configs/my_exp_config.yaml')
parser.add_argument('--exp_model', type=str, default='LLMTime-GPT-3.5', help='LLMTime-GPT-3.5, LLMTime-GPT-4o')
parser.add_argument('--func_name', type=str, default='sigmoid', help='exp, sigmoid, linear_cos')

initial_args, _ = parser.parse_known_args()

# Load configuration from YAML
config_path = initial_args.config
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Set OpenAI API key and base directly from config
openai.api_key = config['common']['openai_api_key']
openai.api_base = config['common']['openai_api_base']

print(os.getcwd())
os.chdir(config['common']['project_root_path'])
print(os.getcwd())

sys.path.append(config['common']['informer_exp_path'])
sys.path.append(config['common']['project_root_path'])

os.environ['OMP_NUM_THREADS'] = config['common']['omp_num_threads']

print(sys.path)

from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data #, get_llmtime_chat_predictions_data
from models.validation_likelihood_tuning import get_autotuned_predictions_data #, get_chat_predictions_data
from data.serialize import SerializerSettings
from data.small_context import get_datasets

from exp_utils import cal_metric, load_pickle, save_pickle, calculate_mean_confidence_interval
from llm_hyperparams import LLM_MODEL_HYPERPARAMS, LLM_PREDICT_FNS
from linear_models.forecasting import train_and_inference

from data_provider.data_factory import data_provider


def plot_preds(train, test, pred_dict, model_name, noise_name='Clean', show_samples=False):

    plt.rcParams['figure.dpi'] = 300 
    google_colors = [
        (140, 26, 245),
        (241, 191, 66),
        (216, 81, 64),
        (88, 166, 92),
        (83, 131, 236),
        (228, 228, 228),
        (183, 183, 183),
        (242, 242, 242),
        (253, 253, 253),
        (255, 255, 255)
    ]
    normalized_colors = [(r/255, g/255, b/255) for r, g, b in google_colors]
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=normalized_colors)
    
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    pred_mse, pred_mae = cal_metric(pred, test)
    plt.figure(figsize=(10, 6))#, dpi=100)
    plt.plot(train, color='black', linewidth=4)
    # plt.plot(test, label=f'Ground Truth ({noise_name})', color='black', linewidth=4)
    plt.plot(test, label=f'Ground Truth', color='black', linewidth=4)
    
    pred_color = normalized_colors[0]
    
    pred_log = f'{model_name}\nMSE:{pred_mse:.6f}\nMAE:{pred_mae:.6f}'
    print(pred_log)
    pred_log = 'Prediction'
    
    plt.plot(pred, color=pred_color, label=pred_log, linewidth=4)
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color=pred_color)
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color=pred_color, alpha=0.3, linewidth=3)
    plt.legend(loc='upper left', fontsize=20)

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # Remove axis labels (if needed)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    plt.gca().spines['top'].set_linewidth(3)
    plt.gca().spines['right'].set_linewidth(3)
    plt.gca().spines['left'].set_linewidth(3)
    plt.gca().spines['bottom'].set_linewidth(3)
    
    plt.savefig(f'{model_name}_{noise_name}.png')



exp_model = initial_args.exp_model

function_data_paths = glob.glob('./experiments_revisiting_llm_acl25/datasets/functions/*')
print(function_data_paths)

llm_model_hypers = LLM_MODEL_HYPERPARAMS[exp_model]
print(llm_model_hypers)
hypers = list(grid_iter(llm_model_hypers))
num_samples = 5


data_dict = dict()
for path in function_data_paths:
    func_ts = np.load(path)[0]
    # Fill NaN values: forward fill, then fill any remaining NaN with 0
    if np.isnan(func_ts).any():
        s = pd.Series(func_ts)
        s = s.fillna(method='ffill').fillna(0)
        func_ts = s.values
    func_ts = func_ts / func_ts.max()
    func_ts = pd.Series(func_ts)
    name = Path(path).stem  # Extract only the file name using pathlib
    data_dict[name] = func_ts

print(data_dict.keys())


data = data_dict[initial_args.func_name]
print(data)

noise_levels = [0.0, 0.001, 0.01, 0.1]
exp_dict = dict()
test_len = int(len(data) * 0.25)

for noise_level in noise_levels[:3]:
    train, test = data.iloc[:-test_len], data.iloc[-test_len:]
    train = train + np.random.normal(0, noise_level, size=len(train))
    
    pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, LLM_PREDICT_FNS[exp_model], verbose=False, parallel=False)
    pred_dict['train_series'] = train
    pred_dict['test_series'] = test
    plot_preds(train, test, pred_dict, exp_model, noise_name=f'{initial_args.func_name}_{noise_level}', show_samples=True)