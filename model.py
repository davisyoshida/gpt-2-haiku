from collections import namedtuple
from functools import partial
import json
from pathlib import Path
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Config = namedtuple('Config', [
    'n_vocab',
    'n_ctx',
    'n_embd',
    'n_head',
    'n_layer',
    'drop_rate',
    'causal_mask'
])

DEFAULT_CONFIG = Config(
    n_vocab=0,
    n_ctx=1024,
    n_embd=768,
    n_head=12,
    n_layer=12,
    drop_rate=0.1,
    causal_mask=True
)

zero_init = hk.initializers.Constant(0)

class ConfigModule(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config

def split_heads(x, n_head):
    # [sequence, features] to [heads, sequence, features]
    m = x.shape[-1]
    split = jnp.reshape(x, x.shape[:-1] + (n_head, m // n_head))
    return jnp.transpose(split, [1, 0, 2])

def merge_heads(x):
    x = jnp.transpose(x, [1, 0, 2])
    *shape, a, b = x.shape
    return jnp.reshape(x, shape + [a * b])

def mask_attn_weights(w, n_past):
    _, nd, ns = w.shape

    i = jnp.arange(nd)[:, None]
    j = jnp.arange(ns)
    b = (i >= j - n_past).astype(w.dtype)
    b = jnp.reshape(b, [1, nd, ns])
    w = w * b - jnp.float32(1e10).astype(w.dtype) * (1 - b)
    return w

class GPT2Attention(ConfigModule):
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.conv1 = hk.Conv1D(config.n_embd * 3, 1, name='c_attn', w_init=hk.initializers.RandomNormal(0.02))
        self.conv2 = hk.Conv1D(config.n_embd, 1, name='c_proj', w_init=hk.initializers.RandomNormal(0.02))

    def __call__(self, x, past, perturb=None, prepend_kv=None, train=False):
        past, past_length = past

        c = self.conv1(x)
        q, k, v = [split_heads(vecs, self.config.n_head) for vecs in jnp.split(c, 3, axis=1)]

        if perturb is not None:
            k += perturb[0]
            v +=  perturb[1]

        if past is not None:
            pk, pv = map(partial(jnp.squeeze, axis=0), jnp.split(past, 2, axis=0))

            k = jax.lax.dynamic_update_slice_in_dim(pk, k, past_length, axis=1)
            v = jax.lax.dynamic_update_slice_in_dim(pv, v, past_length, axis=1)

        if prepend_kv is not None:
            prepend_k, prepend_v = map(partial(jnp.squeeze, axis=0), jnp.split(prepend_kv, 2, axis=0))
            past_length += prepend_k.shape[1]

            k, v = (jnp.concatenate((prefix, vecs), axis=1) for prefix, vecs in [(prepend_k, k), (prepend_v, v)])

        a = self.multihead_attn(q, k, v, train=train, n_past=past_length)
        a = merge_heads(a)
        a = self.conv2(a)
        if train:
            a = hk.dropout(hk.next_rng_key(), self.config.drop_rate, a)
        new_cache = jnp.stack([k, v], axis=0)
        return a, new_cache

    def multihead_attn(self, q, k, v, train=False, n_past=0):
        w = jnp.einsum('hij,hkj->hik', q, k)
        w = w / np.sqrt(v.shape[-1])

        if self.config.causal_mask:
            w = mask_attn_weights(w, n_past=n_past)

        w = jax.nn.softmax(w)
        if train:
            w = hk.dropout(hk.next_rng_key(), self.config.drop_rate, w)
        a = w @ v
        return a

class MLP(ConfigModule):
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.c_fc = hk.Conv1D(4 * config.n_embd, 1, name='c_fc', w_init=hk.initializers.RandomNormal(0.02))
        self.c_proj = hk.Conv1D(config.n_embd, 1, name='c_proj', w_init=hk.initializers.RandomNormal(0.02))

    def __call__(self, h, train=False):
        h = jax.nn.gelu(self.c_fc(h))
        h = self.c_proj(h)
        if train:
            h = hk.dropout(hk.next_rng_key(), self.config.drop_rate, h)
        return h

class GPT2Block(ConfigModule):
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.attn = GPT2Attention(config, name='attn')
        self.mlp = MLP(config, name='mlp')
        self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name='ln_1')
        self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name='ln_2')

    def __call__(self, x, train=False, **kwargs):
        a, present = self.attn(self.norm1(x), train=train, **kwargs)
        x = x + a
        m = self.mlp(self.norm2(x), train=train)
        x = x + m
        return x, present

class GPT2Model(ConfigModule):
    def __init__(self, config, name='model', return_past=False, max_len=None, return_hidden=False):
        super().__init__(config, name)
        self.layers = [GPT2Block(config, name=f'h{i}') for i in range(config.n_layer)]
        self.norm = hk.LayerNorm(-1, create_scale=True, create_offset=True, name='ln_f')
        self.return_past = return_past
        self.return_hidden = return_hidden

        self.max_len = max_len

    def __call__(self, inputs, past, prepend_kv=None, hidden_perturb=None, train=False, use_past=False):
        wte =  hk.get_parameter('wte', [self.config.n_vocab, self.config.n_embd], init=hk.initializers.RandomNormal(0.02))
        wpe =  hk.get_parameter('wpe', [self.config.n_ctx, self.config.n_embd], init=hk.initializers.RandomNormal(0.01))

        if past is None:
            past_length = 0
            if use_past:
                cache_shape = (2, self.config.n_head, self.max_len, self.config.n_embd // self.config.n_head)
                past = [jnp.zeros(cache_shape) for _ in range(self.config.n_layer)]
            else:
                past = [None] * self.config.n_layer
            indices = jnp.arange(inputs.shape[0])
        else:
            past, past_length = past
            indices = jax.lax.dynamic_slice_in_dim(jnp.arange(self.max_len), past_length, inputs.shape[0])

        w_embed = wte[(inputs,)]
        pos_embed = wpe[indices,]
        h = w_embed + pos_embed

        if train:
            h = hk.dropout(hk.next_rng_key(), self.config.drop_rate, h)

        if hidden_perturb is None:
            hidden_perturb = [None] * self.config.n_layer

        if prepend_kv is None:
            prepend_kv = [None] * self.config.n_layer

        presents = []
        hidden = []
        for layer, layer_past, layer_pert, layer_prepend in zip(self.layers, past, hidden_perturb, prepend_kv):
            if self.return_hidden:
                hidden.append(h)
            h, present = layer(h, past=(layer_past, past_length), perturb=layer_pert, prepend_kv=layer_prepend, train=train)
            if self.return_past:
                presents.append(present)
        h = self.norm(h)

        logits = jnp.einsum('te,ve->tv', h, wte)
        ret =  {'logits': logits}
        if self.return_past:
            ret['past'] = (presents, past_length + inputs.shape[0])


        if self.return_hidden:
            hidden.append(h)
            ret['hidden'] = hidden

        return ret

def get_config_and_weights(path):
    path = Path(path)
    with (path / 'config.json').open() as f:
        config = DEFAULT_CONFIG._replace(**json.load(f))

    with (path / 'weights.pkl').open('rb') as f:
        params = pickle.load(f)

    params = jax.tree_map(jnp.array, params)
    n_param = jax.tree_util.tree_reduce(lambda t, a: t + np.prod(a.shape), params, 0)
    print(f'Loaded model with {n_param:.2e} parameters')
    return config, params

def load_model(path='models/117M',
               return_past=False,
               train=False,
               use_past=False,
               return_hidden=False,
               max_len=1024):
    """Load a pretrained GPT-2 model.
    Args:
        path: The directory to load the model from (should include weights.pkl and config.json)
        return_past: Whether to save and return the hidden states
        train: Whether to include dropout
        use_past: Whether to support passing cached hidden states as arguments
        max_len: How large to make the hidden state cache, only used if use_past is True
    """

    model_path = Path(path)
    config, params = get_config_and_weights(model_path)

    def _model_fn(inputs, hidden_perturb=None, past=None, prepend_kv=None):
        return GPT2Model(config, return_past=return_past, max_len=max_len, return_hidden=return_hidden)(inputs, past, hidden_perturb=hidden_perturb, train=train, use_past=use_past, prepend_kv=prepend_kv)

    model = hk.transform(_model_fn)
    if not train:
        model = hk.without_apply_rng(model)

    return model, params
