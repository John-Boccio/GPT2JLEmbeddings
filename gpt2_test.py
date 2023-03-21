import numpy as np
import torch
import transformers
from gpt2_utils import *
import gpt2_jl_embedding
from gpt2_jl_embedding import JLReductionMethod
import amazon_dataset
import itertools
from pathlib import Path
import pandas as pd


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
test_iterations = 1000

model, tokenizer = get_gpt2_model(device=DEVICE)
dataset = amazon_dataset.AmazonBookReviewDataset(tokenizer, batch_size=batch_size, max_length=100, device=DEVICE)

test_results_cols = ['Reduction Method', 'N Components', 'JL Applied', 'Test Acc', '1 Star Acc', '2 Star Acc', '3 Star Acc', '4 Star Acc', '5 Star Acc']
csv_dict = {col: [] for col in test_results_cols}
results_df = pd.DataFrame.from_dict(csv_dict)

pretrained_model_path = Path('gpt2_desc-no-jl_sketch-None_ncomponents-None_batch-8_temp-0.75_lr-0.0001/best.pt')


def model_info_from_name(name: str) -> dict:
    split_name = name.split('_')

    jl_applied = split_name[1].split('-')[1]
    reduction_method = split_name[2].split('-')[1]
    n_components = int(split_name[3].split('-')[1])

    return {
        'Reduction Method': reduction_method,
        'N Components': n_components,
        'JL Applied': jl_applied
    }


def test_model_from_dir(model_dir: Path):
    np.random.seed(269)
    torch.manual_seed(269)

    model_info = model_info_from_name(model_dir.name)
    model, tokenizer = get_gpt2_model(device=DEVICE)

    jl_application = model_info['JL Applied']
    n_components = model_info['N Components']
    reduction_method = JLReductionMethod.__members__[model_info['Reduction Method']]

    best_model_path = model_dir / 'best.pt'
    if jl_application == 'attn':
        gpt2_jl_embedding.apply_jl_gpt2_attention(model, n_components, reduction_method, DEVICE)
    if jl_application == 'conv1d':
        gpt2_jl_embedding.apply_jl_gpt2_conv1d(model, n_components, reduction_method, DEVICE)

    model.load_state_dict(torch.load(best_model_path))

    overall_accuracy, class_accuracy = test_model(model)
    model_info['Test Acc'] = overall_accuracy
    for i in range(5):
        model_info[f'{i+1} Star Acc'] = class_accuracy[i]
    return model_info


def test_model_from_fine_tuned(reduction_method, n_components, jl_application='None'):
    print(f'Testing model with {reduction_method=}, {n_components=}, {jl_application=}')
    np.random.seed(269)
    torch.manual_seed(269)

    model, tokenizer = get_gpt2_model(device=DEVICE)
    model.load_state_dict(torch.load(pretrained_model_path))
    if jl_application == 'attn':
        gpt2_jl_embedding.apply_jl_gpt2_attention(model, n_components, reduction_method, DEVICE)
    if jl_application == 'conv1d':
        gpt2_jl_embedding.apply_jl_gpt2_conv1d(model, n_components, reduction_method, DEVICE)

    overall_accuracy, class_accuracy = test_model(model) 
    
    model_info = {
        'Reduction Method': reduction_method.name if reduction_method is not None else 'None',
        'N Components': n_components if n_components is not None else 0,
        'JL Applied': jl_application,
        'Test Acc': overall_accuracy,
    }
    for i in range(5):
        model_info[f'{i+1} Star Acc'] = class_accuracy[i]

    return model_info


def test_model(model):
    np.random.seed(269)
    torch.manual_seed(269)

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

    return overall_accuracy, class_accuracy


TEST_JL_FROM_TRAINED = True
TEST_JL_FROM_FINE_TUNED = False

if TEST_JL_FROM_TRAINED:
    test_models_summary_path = Path('results/summary_pre_jl.csv')

    models_dir = Path('models')
    for model in models_dir.iterdir():
        model_info = test_model_from_dir(model)
        print(model_info)
        results_df = results_df.append(model_info, ignore_index=True)
        results_df.to_csv(test_models_summary_path)
    
    results_df.to_csv(test_models_summary_path)


if TEST_JL_FROM_FINE_TUNED:
    n_components_attn = [56, 48, 40, 32, 24, 16, 8, 4, 2]
    n_components_conv1d = [672, 576, 480, 384, 288, 192, 96, 48, 24]
    reduction_methods = [gpt2_jl_embedding.JLReductionMethod.GAUSSIAN, gpt2_jl_embedding.JLReductionMethod.SPARSE]

    test_models_summary_path = Path('results/summary_post_jl.csv')

    model_info = test_model_from_fine_tuned(None, None)
    results_df = results_df.append(model_info, ignore_index=True)

    for n_component_conv1d, reduction_method in itertools.product(n_components_conv1d, reduction_methods):
        model_info = test_model_from_fine_tuned(reduction_method, n_component_conv1d, 'conv1d')
        print(model_info)
        results_df = results_df.append(model_info, ignore_index=True)
    
    for n_component_attn, reduction_method in itertools.product(n_components_attn, reduction_methods):
        model_info = test_model_from_fine_tuned(reduction_method, n_component_attn, 'attn')
        print(model_info)
        results_df = results_df.append(model_info, ignore_index=True)

    results_df.to_csv(test_models_summary_path)
