**This repository is the official codebase for the paper ["Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models"](https://arxiv.org/pdf/2506.00457).**

# Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models

This project is based on and extends the work from ["Large Language Models Are Zero Shot Time Series Forecasters"](https://arxiv.org/abs/2310.07820).

## Overview

This repository provides the official implementation and experimental framework for evaluating the zero-shot time series forecasting capabilities of Large Language Models (LLMs) and comparing them to domain-specific and linear models. The codebase supports:
- Zero-shot forecasting with LLMs (e.g., GPT-3.5, GPT-4o, LLaMA)
- Robustness analysis under various noise conditions
- Benchmarking against strong linear and statistical baselines
- Experiments on real-world and synthetic datasets

## Main Features
- **Prompt-based zero-shot forecasting**: Convert time series into prompts for LLMs and evaluate their forecasting ability without domain-specific training.
- **Noise robustness evaluation**: Test model performance under clean, Gaussian, constant, missing, and periodic noise.
- **Comprehensive baselines**: Compare LLMs with domain-specific models (e.g., DLinear, RLinear, ARIMA, N-BEATS).
- **Synthetic and real-world datasets**: Includes experiments on Monash, Informer benchmark datasets, and synthetic mathematical functions.

## Experiments

The main experimental scripts are in the `experiments_revisiting_llm_acl25` directory:

### 1. `run_informer_five_datasets.py`
Benchmarks LLM-based and linear models on five real-world time series datasets (ETTm2, exchange_rate, weather, electricity, traffic). Computes MAE, MSE, and inference time, supporting multiple runs for statistical confidence.

### 2. `run_monash_with_noises.py`
Evaluates model robustness to different types of noise (clean, Gaussian, constant, missing) on the Monash time series datasets. Reports MAE and MSE for each dataset and noise condition.

### 3. `run_function_with_noises.py`
Tests LLM-based models on synthetic mathematical functions (e.g., sigmoid, exp, linear_cos) with varying levels of added noise. Plots and saves predictions with confidence intervals.

### Usage Example

Each script can be run from the command line. Example commands:

```bash
python experiments_revisiting_llm_acl25/run_informer_five_datasets.py --exp_model LLMTime-GPT-3.5 --exp_dataset ETTm2
python experiments_revisiting_llm_acl25/run_monash_with_noises.py --exp_model RLinear --noise_type gaussian
python experiments_revisiting_llm_acl25/run_function_with_noises.py --exp_model LLMTime-GPT-4 --func_name sigmoid
```

Configuration files (YAML) specify dataset paths, OpenAI API keys, and other experiment settings.

## Citation

If you use this code or find our work useful, please cite the following paper:

```
@inproceedings{junwoo2025revisiting,
  title={Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models},
  author={Junwoo Park and Hyuck Lee and Dohyun Lee and Daehoon Gwak and Jaegul Choo},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025},
  url={https://arxiv.org/abs/2506.00457}
}
```
