from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np
from more_itertools import spy

from transformers import GPT2TokenizerFast

def batcher(tokens, batch_size, length=100, random_offset=False, mask=None):
    """Take a bunch of tokens and batch them in order"""
    start = 0
    if random_offset is not None:
        start = np.random.randint(0, length)
    n_tokens = len(tokens) - batch_size - start
    n_tokens = n_tokens - (n_tokens % batch_size)
    end = start + n_tokens

    starts = np.arange(start, end, n_tokens // batch_size)
    while starts[-1] < len(tokens) - 1:
        slices = [tokens[s:s + length + 1] for s in starts]
        if not len(set(map(len, slices))) == 1:
            break
        all_tokens = jnp.array(slices)
        batch = all_tokens[:, :-1], all_tokens[:, 1:]
        if mask is not None:
            batch_mask = jnp.array([mask[s + 1:s + length + 1] for s in starts])
            batch += batch_mask,

        yield batch
        starts += length

def tokenize_file(path, save_tokenized=False):
    """
    Tokenize a file (won't insert <|endoftext|>, so don't use this if that's necessary),
    and optionally save the tokenized data
    """
    path = Path(path)
    pickle_path = path.with_suffix('.tokens.pkl')
    if pickle_path.exists():
        with pickle_path.open('rb') as f:
            return pickle.load(f)

    with path.open() as f:
        tokens = GPT2TokenizerFast.from_pretrained('gpt2').encode(f.read())

    if save_tokenized:
        with pickle_path.open('wb') as f:
            pickle.dump(tokens, f)

    return tokens
