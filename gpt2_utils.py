import torch
import torch.nn as nn
import numpy as np
import tensorboard.summary
import tqdm
import transformers
from pathlib import Path


def get_acc(logits, targets, compute_mean=True):
    if logits.dim() == 2:
        assert logits.dim() == 2
        assert targets.dim() == 1
        assert logits.shape[0] == targets.shape[0]
        y = torch.argmax(logits, dim=-1) == targets
        y = y.type(torch.float)
    elif logits.dim() == 3:
        _, _, vocab_size = logits.shape
        l = logits[:, :-1].view(-1, vocab_size)
        t = targets[:, 1:].view(-1)
        use_indicies = t != -100
        y = torch.argmax(l[use_indicies, :], dim=-1) == t[use_indicies]
        y = y.type(torch.float)
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')
    
    if compute_mean:
        return torch.mean(y).item()
    else:
        return y


def run_validation(model, dataset, iterations=250):
    losses = torch.zeros(iterations * dataset.batch_size)
    accuracies = torch.zeros_like(losses)

    with torch.inference_mode():
        for i in range(iterations): 
            x_, y_ = dataset.get_random_val_batch()       
            logits = model(**x_).logits
            loss = nn.functional.cross_entropy(logits, y_)
            losses[i] = loss.item()
            accuracies[i*dataset.batch_size:(i+1)*dataset.batch_size] = get_acc(logits, y_, compute_mean=False)
    return torch.mean(losses).item(), torch.mean(accuracies).item()


def train_model(model, dataset, log_name, iterations=4000, learning_rate=1e-4):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = tensorboard.summary.Writer(f'logs/{log_name}')

    pbar = tqdm.tqdm(range(iterations))
    best_acc = 0
    for step in pbar:
        x_, y_ = dataset.get_random_train_batch()
        logits = model(**x_).logits
        loss = nn.functional.cross_entropy(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('Accuracy/train', get_acc(logits, y_), step)

        if step % 250 == 0 or step == (iterations-1):
            with torch.inference_mode():
                loss, acc = run_validation(model, dataset)
                Path(f'models/{log_name}/').mkdir(exist_ok=True)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f'models/{log_name}/best.pt')
            writer.add_scalar('Loss/val', loss, step)
            writer.add_scalar('Accuracy/val', acc, step)
            pbar.set_description(f'Fine-tuning loss, acc: {loss:.04f}, {acc:.04f}')


def generate_model_str(model, jl_application, reduction_method=None, n_components=0, batch_size=16, temp=0.75, learning_rate=1e-4):
    reduction_method_name = reduction_method.name if reduction_method is not None else "None"
    return f'{model.name_or_path}_jl-{jl_application}_sketch-{reduction_method_name}_ncomponents-{n_components}_batch-{batch_size}_temp-{temp}_lr-{learning_rate}'


def get_gpt2_model(num_labels=5, temp=0.75, device=None):
    model = transformers.AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels, temperature=temp).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer
