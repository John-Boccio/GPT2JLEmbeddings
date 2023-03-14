import numpy as np
import torch
import transformers
from gpt2_utils import *
import gpt2_jl_embedding
import amazon_dataset
import itertools


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8
learning_rate = 1e-4
n_components_attn = [32, 24, 18, 12, 8, 4]
n_components_conv1d = [384, 288, 192, 96, 64, 32]
reduction_methods = [gpt2_jl_embedding.JLReductionMethod.GAUSSIAN, gpt2_jl_embedding.JLReductionMethod.SPARSE]

TRAIN_NO_JL = True
TRAIN_JL_ATTN = False
TRAIN_JL_CONV1D = False
TRAIN_JL_ATTN_CONV1D = False

model, tokenizer = get_gpt2_model(device=DEVICE)
dataset = amazon_dataset.AmazonBookReviewDataset(tokenizer, batch_size=batch_size, max_length=100, device=DEVICE)

if TRAIN_NO_JL:
    print('TRAIN_NO_JL')
    model, tokenizer = get_gpt2_model(device=DEVICE)
    log_name = generate_model_str(model, 'no-jl', batch_size=batch_size, learning_rate=learning_rate)
    print(f'Training model {log_name}')
    train_model(model, dataset, log_name, learning_rate=learning_rate)

if TRAIN_JL_ATTN:
    print('TRAIN_JL_ATTN')
    # Training with JL embeddings applied to attention computation
    for n_component_attn, reduction_method in itertools.product(n_components_attn, reduction_methods):
        model, tokenizer = get_gpt2_model(device=DEVICE)
        log_name = generate_model_str(model, 'jl-attn', reduction_method=reduction_method, n_components=n_component_attn, batch_size=batch_size, learning_rate=learning_rate)
        gpt2_jl_embedding.apply_jl_gpt2_attention(model, n_component_attn, reduction_method, DEVICE)
        print(f'Training JL attn model {log_name}')
        train_model(model, dataset, log_name, learning_rate=learning_rate)

if TRAIN_JL_CONV1D:
    print('TRAIN_JL_CONV1D')
    # Training with JL embeddings applied to conv1d computation
    for n_component_conv1d, reduction_method in itertools.product(n_components_conv1d, reduction_methods):
        model, tokenizer = get_gpt2_model(device=DEVICE)
        log_name = generate_model_str(model, 'jl-conv1d', reduction_method=reduction_method, n_components=n_component_conv1d, batch_size=batch_size, learning_rate=learning_rate)
        gpt2_jl_embedding.apply_jl_gpt2_conv1d(model, n_component_conv1d, reduction_method, DEVICE)
        print(f'Training JL conv1d model {log_name}')
        train_model(model, dataset, log_name, learning_rate=learning_rate)

if TRAIN_JL_ATTN_CONV1D:
    print('TRAIN_JL_ATTN_CONV1D')
    # Training with JL embeddings applied to both conv1d and attnetntion computation
    for i in range(n_components_attn):
        n_component_attn = n_components_attn[i]
        n_component_conv1d = n_components_conv1d[i]
        for reduction_method in reduction_methods:
            model, tokenizer = get_gpt2_model(device=DEVICE)
            log_name = generate_model_str(model, 'jl-conv1d-attn', reduction_method=reduction_method, n_components=(n_component_conv1d, n_component_attn), batch_size=batch_size, learning_rate=learning_rate)
            gpt2_jl_embedding.apply_jl_gpt2_conv1d(model, n_component_conv1d, reduction_method, DEVICE)
            gpt2_jl_embedding.apply_jl_gpt2_attention(model, n_component_attn, reduction_method, DEVICE)
            print(f'Training JL conv1d model {log_name}')
            train_model(model, dataset, log_name, learning_rate=learning_rate)
