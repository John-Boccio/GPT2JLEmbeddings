import numpy as np
import torch
import transformers
from gpt2_utils import *
import gpt2_jl_embedding
import amazon_dataset
import itertools
from pathlib import Path
import pandas as pd


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
test_iterations = 1000

n_components_attn = [56, 48, 40, 32, 24, 16, 8, 4, 2]
n_components_conv1d = [672, 576, 480, 384, 288, 192, 96, 48, 24]
reduction_methods = [gpt2_jl_embedding.JLReductionMethod.GAUSSIAN, gpt2_jl_embedding.JLReductionMethod.SPARSE]

model, tokenizer = get_gpt2_model(device=DEVICE)
dataset = amazon_dataset.AmazonBookReviewDataset(tokenizer, batch_size=batch_size, max_length=100, device=DEVICE)

models_dir = Path('models')

test_results_cols = ['Reduction Method', 'N Components', 'JL Applied', 'Quantized', 'Test Acc', '1 Star Acc', '2 Star Acc', '3 Star Acc', '4 Star Acc', '5 Star Acc']
csv_dict = {col: [] for col in test_results_cols}
results_df = pd.DataFrame.from_dict(csv_dict)

best_model_path = Path('gpt2_desc-no-jl_sketch-None_ncomponents-None_batch-8_temp-0.75_lr-0.0001/best.pt')


def test_model(reduction_method, n_components, jl_application='None', quantization='None'):
    print(f'Testing model with {reduction_method=}, {n_components=}, {jl_application=}, {quantization=}')
    np.random.seed(269)
    torch.manual_seed(269)

    model, tokenizer = get_gpt2_model(device=DEVICE)
    model.load_state_dict(torch.load(best_model_path))
    if jl_application == 'attn':
        gpt2_jl_embedding.apply_jl_gpt2_attention(model, n_components, reduction_method, DEVICE)
    if jl_application == 'conv1d':
        gpt2_jl_embedding.apply_jl_gpt2_conv1d(model, n_components, reduction_method, DEVICE)
    
    if quantization != 'None':
        assert False

    with torch.inference_mode():
        pbar = tqdm.tqdm(range(test_iterations))
        y_preds = torch.zeros(test_iterations * batch_size)
        y_actuals = torch.zeros(test_iterations * batch_size)
        for step in pbar:
            x_, y_actual = dataset.get_random_test_batch()
            loss, logits = model(**x_, labels=y_actual)[:2]
            y_pred = torch.argmax(logits, dim=-1)

            y_actuals[step*batch_size:(step+1)*batch_size] = y_actual
            y_preds[step*batch_size:(step+1)*batch_size] = y_pred

    overall_accuracy = torch.mean((y_actuals == y_preds).type(torch.float)).item()
    class_accuracy = [0]*5
    for i in range(5):
        class_idxs = (y_actuals == i)
        class_accuracy[i] = torch.mean((y_preds[class_idxs] == i).type(torch.float)).item()

    model_info = {
        'Reduction Method': reduction_method.name if reduction_method is not None else 'None',
        'N Components': n_components if n_components is not None else 0,
        'JL Applied': jl_application,
        'Quantized':quantization,
        'Test Acc': overall_accuracy,
    }
    for i in range(5):
        model_info[f'{i+1} Star Acc'] = class_accuracy[i]

    return model_info


TEST_POST_JL = True

if TEST_POST_JL:
    test_models_summary_path = models_dir / 'summary_post_jl.csv'

    model_info = test_model(None, None)
    results_df = results_df.append(model_info, ignore_index=True)

    for n_component_conv1d, reduction_method in itertools.product(n_components_conv1d, reduction_methods):
        model_info = test_model(reduction_method, n_component_conv1d, 'conv1d')
        results_df = results_df.append(model_info, ignore_index=True)
    
    for n_component_attn, reduction_method in itertools.product(n_components_attn, reduction_methods):
        model_info = test_model(reduction_method, n_component_attn, 'attn')
        results_df = results_df.append(model_info, ignore_index=True)

    results_df.to_csv(test_models_summary_path)
