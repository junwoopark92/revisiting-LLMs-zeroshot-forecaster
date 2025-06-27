from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data 

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0, # 1.0
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-') # 3
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)


promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)

arima_hypers = dict(p=[12,30], d=[1,2], q=[0])

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

LLM_MODEL_HYPERPARAMS = {
    'LLMTime-GPT-4o' :{'model': 'gpt-4o', **gpt4_hypers},
    'LLMTime-GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
    # 'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo', **gpt3_hypers},
    'LLMTime-GPT-4': {'model': 'gpt-4', **gpt4_hypers},
    # 'LLMTime GPT-3': {'model': 'text-davinci-003', **gpt3_hypers},
    'LLMA2-7b': {'model': 'llama-7b', **llama_hypers},
    'LLMA2-70b': {'model': 'llama-70b', **llama_hypers},
    'LLMA3-8b': {'model': 'llama3-8b', **llama_hypers},
    'LLMA3-8b chat': {'model': 'llama3-8b-chat', **llama_hypers},
    'LLMA3-70b chat': {'model': 'llama3-70b-chat', **llama_hypers},
    'LLMA3.1-8b': {'model': 'llama3.1-8b', **llama_hypers},
    'LLMA3.1-8b chat': {'model': 'llama3.1-8b-chat', **llama_hypers},
    'LLMA3.3-70b chat': {'model': 'llama3.3-70b-chat', **llama_hypers},
}

LLM_PREDICT_FNS = {
    'LLMTime-GPT-3.5': get_llmtime_predictions_data,
    'LLMTime-GPT-4': get_llmtime_predictions_data,
    'LLMA2-70b': get_llmtime_predictions_data,
    'LLMA2-7b': get_llmtime_predictions_data,
    'LLMA3-8b': get_llmtime_predictions_data,
    'LLMA3.1-8b': get_llmtime_predictions_data,
    # 'LLMA3-70b-chat': get_llmtime_chat_predictions_data,
    # 'LLMA3-8b-chat': get_llmtime_chat_predictions_data,
    # 'LLMA3.1-8b-chat': get_llmtime_chat_predictions_data,
    # 'LLMA3.3-70b-chat': get_llmtime_chat_predictions_data,
    'LLMTime-GPT-4o': get_llmtime_predictions_data,
    # 'PromptCast GPT-3': get_promptcast_predictions_data,
    # 'ARIMA': get_arima_predictions_data,
}