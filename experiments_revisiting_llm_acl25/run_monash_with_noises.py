import os
import sys
import time
import yaml

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import argparse

from ts_argparse import informer_parser


# Initial parsing of arguments to get chosen_model and chosen_dataset
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='experiments_revisiting_llm_acl25/configs/my_exp_config.yaml')
parser.add_argument('--exp_model', type=str, default='LLMTime-GPT-3.5', help='RLinear, LLMTime-GPT-3.5, LLMTime-GPT-4o')
parser.add_argument('--noise_type', type=str, default='clean', help='clean, gaussian constant missing')

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



exp_model = initial_args.exp_model

datasets = get_datasets()
ds_names = list(datasets.keys())
for k, v in datasets.items():
    print(k, v[0].shape[0] + v[1].shape[0])


if exp_model in ['DLinear', 'NLinear', 'RLinear']:
    learning_rate = 0.0001
    if exp_model == 'DLinear':
        from linear_models.DLinear import Model
    elif exp_model == 'NLinear':
        from linear_models.NLinear import Model    
    elif exp_model == 'RLinear':
        from linear_models.RLinear import Model
    else:
        raise Exception(f'Unsupported model: {exp_model}')
    
else:
    llm_model_hypers = LLM_MODEL_HYPERPARAMS[exp_model]
    print(llm_model_hypers)
    hypers = list(grid_iter(llm_model_hypers))
    num_samples = 5


all_mses = []
all_maes = []
for ds_name in ds_names:
    data = datasets[ds_name]
    train, test = data # or change to your own data
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean)/train_std
    train_len = len(train)
    noise_len = int(train_len*0.2)

    if initial_args.noise_type == 'gaussian':
        random_index = np.random.choice(train_len, noise_len, replace=True)
        train.iloc[random_index] = train.iloc[random_index] + np.random.normal(0.0, 1.0, size=noise_len)
    elif initial_args.noise_type == 'constant':
        random_index = np.random.choice(train_len,noise_len, replace=True)
        train.iloc[random_index] = train.iloc[random_index] + 0.5
    elif initial_args.noise_type == 'missing':
        random_index = np.random.choice(train_len,noise_len, replace=True)
        train.iloc[random_index] = 0.5
    elif initial_args.noise_type == 'clean':
        pass
    else:
        raise Exception
    
    test = (test - train_mean)/train_std

    if exp_model in ['DLinear', 'RLinear', 'NLinear']:
        internal_seq_len, internal_pred_len = len(test), len(test)
        train = train.values.reshape(-1, 1)
        test = test.values.reshape(-1, 1)

        pred_seq = train_and_inference(Model, train, test, internal_seq_len, internal_pred_len, stride=internal_seq_len,
                                        epochs=10, learning_rate=learning_rate, decom_len=5, train_ratio=0.85)[0]
        print(pred_seq.shape, test.shape)
        sample_mse, sample_mae = cal_metric(pred_seq, test)
    else:
        pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, LLM_PREDICT_FNS[exp_model], verbose=False, parallel=False)
        pred_dict['train_series'] = train
        pred_dict['test_series'] = test
        sample_mse, sample_mae = cal_metric(pred_dict['median'], pred_dict['test_series'])
    
    print(f"[{ds_name}] MSE:{sample_mse:.4f} MAE:{sample_mae:.4f}")

    all_mses.append(sample_mse)
    all_maes.append(sample_mae)

last_mse_mean, last_mse_margin = calculate_mean_confidence_interval(all_mses)
last_mae_mean, last_mae_margin = calculate_mean_confidence_interval(all_maes)

print("\n==================== Forecasting Results Summary ====================")
print(f"Model: {exp_model}")
print(f"Dataset: Monash with {initial_args.noise_type}")
print("---------------------------------------------------------------------")
print(f"{'Metric':<20}{'Mean':>15}")
print("---------------------------------------------------------------------")
print(f"{'MAE':<20}{last_mae_mean:>15.4f}")
print(f"{'MSE':<20}{last_mse_mean:>15.4f}")
print("=====================================================================\n")