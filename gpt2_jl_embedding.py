import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model, GPT2Model
import transformers
import torch.nn as nn
from typing import Optional, Tuple, Union
from sklearn.random_projection import _gaussian_random_matrix, _sparse_random_matrix
from enum import Enum


class JLReductionMethod(Enum):
    GAUSSIAN = 0
    SPARSE = 1
    LEARNED = 2

    def generate_proj_matrix(self, n_components, n_features, device):
        if self == self.GAUSSIAN or self == self.LEARNED:
            S = torch.tensor(_gaussian_random_matrix(n_components, n_features), dtype=torch.float, device=device)
        elif self == self.SPARSE:
            S = torch.tensor(_sparse_random_matrix(n_components, n_features).todense(), dtype=torch.float, device=device)
        return S


class JLConv1D(nn.Module):
    def __init__(self, conv1d, n_components, reduction_method, device, training=False) -> None:
        super().__init__()
        self.conv1d = conv1d
        self.n_components = n_components
        self.reduction_method = reduction_method
        self.training = training
        self.device = device
        if n_components >= self.conv1d.weight.shape[0]:
            raise ValueError(f"Num. components {n_components} is greater than the number of features {self.conv1d.nf}")
        self.S = reduction_method.generate_proj_matrix(n_components, self.conv1d.weight.shape[0], device)
        if self.reduction_method == JLReductionMethod.LEARNED:
            self.S = nn.parameter.Parameter(self.S, requires_grad=True)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.conv1d.nf,)
        # Same as the original conv1d, but applying the dimensionality reduction
        if self.training and self.reduction_method != JLReductionMethod.LEARNED:
            self.S = self.reduction_method.generate_proj_matrix(self.n_components, self.conv1d.weight.shape[0], self.device)
        x = torch.addmm(self.conv1d.bias, x.view(-1, x.size(-1)) @ self.S.T, self.S @ self.conv1d.weight)
        x = x.view(*size_out)
        return x


class JLGPT2Attention(nn.Module):
    def __init__(self, gpt2_attention: GPT2Attention, n_components, reduction_method, device, training=False) -> None:
        super().__init__()
        self.gpt2_attention = gpt2_attention
        self.n_components = n_components
        self.reduction_method = reduction_method
        self.training = training
        self.device = device
        if n_components >= self.gpt2_attention.head_dim:
            raise ValueError(f"Num. components {n_components} is greater than the attention head dimension {self.gpt2_attention.head_dim}")
        self.S = reduction_method.generate_proj_matrix(n_components, self.gpt2_attention.head_dim, device)
        if self.reduction_method == JLReductionMethod.LEARNED:
            self.S = nn.parameter.Parameter(self.S, requires_grad=True)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """
        This is taken from https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/gpt2/modeling_gpt2.py#L299

        The only difference is that we are applying the JL embedding to the query, key, and value matrices for 
        dimensionality reduction.
        """
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.gpt2_attention.q_attn(hidden_states)
            key, value = self.gpt2_attention.c_attn(encoder_hidden_states).split(self.gpt2_attention.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.gpt2_attention.c_attn(hidden_states).split(self.gpt2_attention.split_size, dim=2)

        if self.training and self.reduction_method != JLReductionMethod.LEARNED:
            self.S = self.reduction_method.generate_proj_matrix(self.n_components, self.gpt2_attention.head_dim, self.device)

        query = self.gpt2_attention._split_heads(query, self.gpt2_attention.num_heads, self.gpt2_attention.head_dim) @ self.S.T
        key = self.gpt2_attention._split_heads(key, self.gpt2_attention.num_heads, self.gpt2_attention.head_dim) @ self.S.T
        value = self.gpt2_attention._split_heads(value, self.gpt2_attention.num_heads, self.gpt2_attention.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.gpt2_attention.reorder_and_upcast_attn:
            attn_output, attn_weights = self.gpt2_attention._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self.gpt2_attention._attn(query, key, value, attention_mask, head_mask)

        attn_output = self.gpt2_attention._merge_heads(attn_output, self.gpt2_attention.num_heads, self.gpt2_attention.head_dim)
        attn_output = self.gpt2_attention.c_proj(attn_output)
        attn_output = self.gpt2_attention.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def apply_jl_gpt2_attention(model: GPT2Model, n_components, reduction_method, device, training=False, count=None):
    if count is None:
        count = len(model.transformer.h)
    model.transformer.wte.requires_grad = False
    model.transformer.wpe.requires_grad = False
    for idx, gpt2_block in enumerate(model.transformer.h): #type: GPT2Block
        gpt2_block.attn = JLGPT2Attention(gpt2_block.attn, n_components, reduction_method, device, training)
        if idx < count:
            gpt2_block.requires_grad = False


def apply_jl_gpt2_conv1d(model: GPT2Model, n_components, reduction_method, device, training=False, count=None):
    if count is None:
        count = len(model.transformer.h)
    model.transformer.wte.requires_grad = False
    model.transformer.wpe.requires_grad = False
    for idx, gpt2_block in enumerate(model.transformer.h):
        gpt2_block.attn.c_attn = JLConv1D(gpt2_block.attn.c_attn, n_components, reduction_method, device, training)
        if idx < count:
            gpt2_block.requires_grad = False


def compute_jl_gpt2_attention_flops_savings(model: GPT2Model, n_components, tokens=256):
    flops_savings = 0
    for gpt2_block in model.transformer.h: #type: GPT2Block
        flops = gpt2_block.attn.num_heads * tokens * gpt2_block.attn.head_dim * tokens
        new_flops = gpt2_block.attn.num_heads * tokens * n_components * tokens + 2 * tokens * gpt2_block.attn.head_dim * n_components
        flops_savings += (flops - new_flops)
    return flops_savings


def compute_jl_gpt2_conv1d_flops_savings(model: GPT2Model, n_components, tokens=256):
    flops_savings = 0
    for gpt2_block in model.transformer.h: #type: GPT2Block
        embed_dim = gpt2_block.attn.embed_dim
        flops = tokens * embed_dim * embed_dim*3
        new_flops = tokens * n_components * embed_dim*3 + tokens * embed_dim * n_components + n_components * embed_dim * embed_dim*3
        flops_savings += (flops - new_flops)
    return flops_savings
