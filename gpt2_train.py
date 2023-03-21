import numpy as np
import torch
import transformers
from gpt2_utils import *
from gpt2_jl_embedding import *
import amazon_dataset
import itertools


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8
learning_rate = 1e-4
n_components_attn = [56, 48, 40, 32, 24, 16, 8, 4, 2]
n_components_conv1d = [672, 576, 480, 384, 288, 192, 96, 48, 24]
reduction_methods = [JLReductionMethod.LEARNED, JLReductionMethod.GAUSSIAN, JLReductionMethod.SPARSE]

TRAIN_NO_JL = False
TRAIN_JL_ATTN = True
TRAIN_JL_CONV1D = True

model, tokenizer = get_gpt2_model(device=DEVICE)
dataset = amazon_dataset.AmazonBookReviewDataset(tokenizer, batch_size=batch_size, max_length=100, device=DEVICE)

if TRAIN_NO_JL:
    print('TRAIN_NO_JL')
    model, tokenizer = get_gpt2_model(device=DEVICE)
    log_name = generate_model_str(model, 'None', batch_size=batch_size, learning_rate=learning_rate)
    print(f'Training model {log_name}')
    train_model(model, dataset, log_name, learning_rate=learning_rate)

if TRAIN_JL_CONV1D:
    print('TRAIN_JL_CONV1D')
    # Training with JL embeddings applied to conv1d computation
    for n_component_conv1d, reduction_method in itertools.product(n_components_conv1d, reduction_methods):
        model, tokenizer = get_gpt2_model(device=DEVICE)
        log_name = generate_model_str(model, 'conv1d', reduction_method=reduction_method, n_components=n_component_conv1d, batch_size=batch_size, learning_rate=learning_rate)
        apply_jl_gpt2_conv1d(model, n_component_conv1d, reduction_method, DEVICE, training=True)
        print(f'Training JL conv1d model {log_name}')
        train_model(model, dataset, log_name, learning_rate=learning_rate)

if TRAIN_JL_ATTN:
    print('TRAIN_JL_ATTN')
    # Training with JL embeddings applied to attention computation
    for n_component_attn, reduction_method in itertools.product(n_components_attn, reduction_methods):
        model, tokenizer = get_gpt2_model(device=DEVICE)
        log_name = generate_model_str(model, 'attn', reduction_method=reduction_method, n_components=n_component_attn, batch_size=batch_size, learning_rate=learning_rate)
        apply_jl_gpt2_attention(model, n_component_attn, reduction_method, DEVICE, training=True)
        print(f'Training JL attn model {log_name}')
        train_model(model, dataset, log_name, learning_rate=learning_rate)
