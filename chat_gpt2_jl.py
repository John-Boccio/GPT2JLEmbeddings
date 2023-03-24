import transformers
from gpt2_jl_embedding import *
import torch


# https://huggingface.co/blog/how-to-generate

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-xl")
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-xl", pad_token_id=tokenizer.eos_token_id)

n_components = 24
reduction_method = JLReductionMethod.GAUSSIAN

# Optionally apply the approximate matrix multiplications to the model before prompting it
# apply_jl_gpt2_conv1d(model, n_components, reduction_method, device=None)
apply_jl_gpt2_attention(model, n_components, reduction_method, device=None)

with torch.inference_mode():
    print('Enter \"exit\" at any point to exit')
    while (prompt := input('Enter prompt: ')).lower() != 'exit':
        output = model.generate(
            tokenizer.encode(prompt, return_tensors='pt').to(),
            max_length=50, 
            num_beams=5,
            no_repeat_ngram_size=2, 
            early_stopping=True
        )
    
        print(f'Output:\n-------')
        print(tokenizer.decode(output[0], skip_special_tokens=True))
