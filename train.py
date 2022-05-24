from collections import deque
from functools import partial
from pathlib import Path
import pickle
import shutil

import haiku as hk
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import numpy as np
import pickle
from tqdm import tqdm

from .model import load_model

def _wrap_generator(gen):
    for batch, targets, *mask in gen:
        mask = mask[0] if mask else None
        yield batch, targets, mask

def evaluate_model(data_gen, loss_fn, params):
    bar = tqdm(_wrap_generator(data_gen()), desc='Evaluating', dynamic_ncols=True)
    total_loss = 0
    tokens = 0
    for batch, targets, mask in bar:
        total_loss += loss_fn(params, batch, targets, mask)
        tokens += jnp.sum(mask)

    ppl = jnp.exp(total_loss / tokens)
    return ppl


def train_model(
        batch_gen,
        val_gen=None,
        epochs=2,
        init_path='models/117M',
        out_path='trained',
        out_name='weights',
        learning_rate=1e-5,
        warmup_steps=300):
    """Train a GPT-2 model
    Args:
        batch_gen: A callable which returns an iterator yielding batches
        init_path: Model to initialize from
        out_path: Directory to save result to
        save_period: Save every `save_period` epochs
    """

    eval_model, _ = load_model(init_path, train=False)
    del _

    model, params = load_model(init_path, train=True)

    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir()
    save_file = out_path / f'{out_name}.pkl'
    shutil.copy(
        Path(init_path) / 'config.json',
        Path(out_path) / 'config.json'
    )

    def loss_func(params, inputs, targets, mask, rng):
        if rng is None:
            output = eval_model.apply(params, inputs=inputs)
        else:
            output = model.apply(params, inputs=inputs, rng=rng)
        logits = output['logits']
        token_losses = jnp.take_along_axis(-jax.nn.log_softmax(logits), targets[:, None], axis=1)
        if mask is not None:
            token_losses = jnp.squeeze(token_losses, axis=1)
            token_losses = token_losses * mask

        return jnp.sum(token_losses)

    def batch_loss_func(params, inputs, targets, mask, rng_key):
        keys = None if rng_key is None else jax.random.split(rng_key, inputs.shape[0])
        return jnp.sum(jax.vmap(loss_func, in_axes=(None, 0, 0, 0, 0))(params, inputs, targets, mask, keys))

    eval_loss_fn = jax.jit(partial(batch_loss_func, rng_key=None))

    def lr_schedule(t):
        warmup_scale = jax.lax.select(t <= warmup_steps, t / warmup_steps, 1.)
        return learning_rate * warmup_scale

    opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
    opt_state = opt_init(params)
    del params

    def update(opt_state, t, batch_inputs, batch_targets, mask, rng_key):
        loss, (grads,) = jax.value_and_grad(batch_loss_func, argnums=(0,))(
            get_params(opt_state),
            batch_inputs,
            batch_targets,
            mask,
            rng_key
        )
        return opt_update(t, grads, opt_state), loss

    update_fn = jax.jit(update, donate_argnums=(0,))

    t = jnp.zeros(())
    step = 0

    last_epoch_loss = float('inf')
    best_ppl = float('inf')

    key = jax.random.PRNGKey(0)
    for epoch in range(1, epochs + 1):
        loss_deque = deque(maxlen=50)
        bar = tqdm(
            _wrap_generator(batch_gen()),
            dynamic_ncols=True
        )

        epoch_loss = 0
        epoch_start = step
        for batch, targets, mask in bar:
            key, = jax.random.split(key, 1)
            t = t + 1
            opt_state, loss = update_fn(opt_state, t, batch, targets, mask, key)
            if jnp.isnan(loss):
                breakpoint()
            loss_deque.append(loss)
            epoch_loss += loss
            if step % 50 == 0:
                avg_loss = np.mean([l.item() for l in loss_deque])
                eval_str = '' if val_gen is None else f' Best val perplexity: {best_ppl:.2f}'
                bar.set_description(f'Epoch {epoch}, Step {step} - Average loss: {avg_loss:0.2e} (Last epoch loss: {last_epoch_loss:0.2e}){eval_str}')
            step += 1

        last_epoch_loss = epoch_loss.item() / (step - epoch_start)

        if val_gen is not None:
            ppl = evaluate_model(val_gen, eval_loss_fn, get_params(opt_state))
            if ppl < best_ppl:
                best_ppl = ppl
                with save_file.open('wb') as f:
                    pickle.dump(get_params(opt_state), f)

    if val_gen is None:
        with save_file.open('wb') as f:
            pickle.dump(get_params(opt_state), f)
