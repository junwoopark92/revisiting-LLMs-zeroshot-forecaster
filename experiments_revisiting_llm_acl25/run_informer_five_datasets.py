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
parser.add_argument('--exp_model', type=str, default='RLinear', help='RLinear, LLMTime-GPT-3.5, LLMTime-GPT-4o')
parser.add_argument('--exp_dataset', type=str, default='ETTm2', help='')
parser.add_argument('--n_runs', type=int, default=1, help='')

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
exp_dataset = initial_args.exp_dataset

dataset_params = config['datasets'][exp_dataset]
seq_len = dataset_params['seq_len']
pred_len = dataset_params['pred_len']
dataset_name = dataset_params['dataset_name']
n_features = dataset_params['n_features']
data_type = dataset_params['data_type']

command=f"""
--task_name long_term_forecast 
--is_training 1 
--root_path {os.path.join(config['common']['informer_exp_path'], 'datasets', 'informer')} 
--data_path {dataset_name}
--model_id 'zeroshot'
--model {exp_model}
--data {data_type} 
--features M 
--seq_len {seq_len}
--label_len 10
--pred_len {pred_len} 
--e_layers 2 
--d_layers 1 
--factor 3 
--enc_in {n_features}
--c_out {n_features}
--dec_in {n_features}
--expand 2 
--d_ff 64 
--d_conv 4  
--d_model 32 
--des 'Exp' 
--itr 5 
--train_epochs 3
--learn None
--use_label_method None
"""
args = informer_parser.parse_args(command.split())
print(args)

data_set, data_loader = data_provider(args, 'test')
index, batch_x, batch_y, batch_x_mark, batch_y_mark, _ = data_set[len(data_set)-1]
all_train = batch_x
all_test = batch_y[-pred_len:]
_, n_channels = all_train.shape
print(all_train.shape, all_test.shape)

# Inference
if exp_model in ['DLinear', 'NLinear', 'RLinear']:
    learning_rate = 0.005
    if exp_model == 'DLinear':
        from linear_models.DLinear import Model
    elif exp_model == 'NLinear':
        from linear_models.NLinear import Model    
    elif exp_model == 'RLinear':
        from linear_models.RLinear import Model
    else:
        raise Exception(f'Unsupported model: {exp_model}')

    train_len = len(all_train)

    if train_len <= pred_len*2:
        internal_seq_len = pred_len//2
    else:
        internal_seq_len = pred_len
        
    internal_pred_len = internal_seq_len
    print(internal_seq_len, internal_seq_len)

    all_mses = []
    all_maes = []
    inference_times = []
    for n_iter in range(initial_args.n_runs):
        train, test = all_train, all_test
        st_time = time.time()
        pred_seq = train_and_inference(Model, all_train, all_test, internal_seq_len, internal_pred_len, internal_seq_len//2, epochs=10, learning_rate=learning_rate)[0]
        ed_time = time.time()
        
        infer_mse, infer_mae = cal_metric(pred_seq, all_test)
        inference_time = ed_time - st_time
        inference_times.append(inference_time)
        
        print(f'{infer_mae:.4f}\t\t{infer_mse:.4f}')
        all_mses.append(infer_mse)
        all_maes.append(infer_mae)

        print(f'MAE:{infer_mae:.4f}\tMSE:{infer_mse:.4f}\tTIME:{inference_time:.4f}')

else:
    llm_model_hypers = LLM_MODEL_HYPERPARAMS[exp_model]
    print(llm_model_hypers)

    all_mses = []
    all_maes = []
    inference_times = []
    for n_iter in range(initial_args.n_runs):
        out = {}
        out[exp_model] = []
        hypers = list(grid_iter(llm_model_hypers))
        num_samples = 20

        st_time = time.time()
        for channel_idx in range(n_channels):
            train, test = pd.Series(all_train[:, channel_idx], index=range(seq_len)), pd.Series(all_test[:, channel_idx], index=range(seq_len, seq_len + pred_len))
            pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, LLM_PREDICT_FNS[exp_model], verbose=False, parallel=False)
            pred_dict['train_series'] = train
            pred_dict['test_series'] = test
            if (n_channels < 25) or (channel_idx % 20 == 0):
                sample_mse, sample_mae = cal_metric(pred_dict['median'], pred_dict['test_series'])
                print(f"[{channel_idx}/{n_channels}] MSE:{sample_mse:.4f} MAE:{sample_mae:.4f}")
            out[exp_model].append(pred_dict)
        inference_time = time.time() - st_time

        # Aggregate Metrics
        all_pred = []
        ignore_index = []
        for channel_idx, pred_per_channel in enumerate(out[exp_model]):
            if pred_per_channel is None:
                ignore_index.append(channel_idx)
            else:
                all_pred.append(pred_per_channel['median'])

        all_pred = pd.concat(all_pred, axis=1).values
        print(ignore_index, all_pred.shape)

        mse, mae = cal_metric(all_pred, np.delete(all_test, ignore_index, axis=1))
        all_mses.append(mse)
        all_maes.append(mae)
        inference_times.append(inference_time)
        print(f'MAE:{mae:.4f}\tMSE:{mse:.4f}\tTIME:{inference_time:.4f}')


last_mse_mean, last_mse_margin = calculate_mean_confidence_interval(all_mses)
last_mae_mean, last_mae_margin = calculate_mean_confidence_interval(all_maes)
inference_time_mean, inference_time_margin = calculate_mean_confidence_interval(inference_times)

print("\n==================== Forecasting Results Summary ====================")
print(f"Model: {exp_model}")
print(f"Dataset: {exp_dataset}")
print(f"Runs: {initial_args.n_runs}")
print("---------------------------------------------------------------------")
print(f"{'Metric':<20}{'Mean':>15}{'95% CI Margin':>20}")
print("---------------------------------------------------------------------")
print(f"{'MAE':<20}{last_mae_mean:>15.4f}{last_mae_margin:>20.4f}")
print(f"{'MSE':<20}{last_mse_mean:>15.4f}{last_mse_margin:>20.4f}")
print(f"{'Inference Time (s)':<20}{inference_time_mean:>15.4f}{inference_time_margin:>20.4f}")
print("=====================================================================\n")