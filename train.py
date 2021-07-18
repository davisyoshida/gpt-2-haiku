from collections import deque
from pathlib import Path
import pickle

import jax
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from data import batcher, tokenize_file
from model import load_model

def main(data_file,
         epochs=2,
         init_path='models/117M',
         out_path='trained',
         save_period=None,
         batch_size=8,
         learning_rate=1e-5,
         ex_len=1024,
         warmup_steps=300):
    """Train a GPT-2 model
    Args:
        data_file: A text file to train from
        init_path: Model to initialize from
        out_path: Directory to save result to
        save_period: Save every `save_period` epochs
    """

    model, params = load_model(init_path, train=True)

    tokens = tokenize_file(data_file)
    steps_per_epoch = len(tokens) // (ex_len * batch_size)

    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir()

    def loss_func(params, inputs, targets, rng):
        output = model.apply(params, inputs=inputs, rng=rng)
        logits = output['logits']
        token_losses = jnp.take_along_axis(-jax.nn.log_softmax(logits), targets[:, None], axis=1)
        return jnp.mean(token_losses)

    def batch_loss_func(params, inputs, targets, rng_key):
        keys = jax.random.split(rng_key, inputs.shape[0])
        return jnp.mean(jax.vmap(loss_func, in_axes=(None, 0, 0, 0))(params, inputs, targets, keys))

    def lr_schedule(t):
        n_steps = (epochs * len(tokens) // (batch_size * ex_len)) - warmup_steps
        warmup_scale = jax.lax.select(t <= warmup_steps, t / warmup_steps, 1.)
        anneal_amount = jax.lax.select(t <= warmup_steps, 0., (t - warmup_steps) / n_steps)
        return learning_rate * warmup_scale * (1 - anneal_amount)

    opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
    opt_state = opt_init(params)
    del params

    def update(opt_state, t, batch_inputs, batch_targets, rng_key):
        loss, (grads,) = jax.value_and_grad(batch_loss_func, argnums=(0,))(get_params(opt_state), batch_inputs, batch_targets, rng_key)
        return opt_update(t, grads, opt_state), loss

    update_fn = jax.jit(update, donate_argnums=(0,))

    t = jnp.zeros(())
    step = 0

    last_epoch_loss = float('inf')

    key = jax.random.PRNGKey(0)
    for epoch in range(1, epochs + 1):
        loss_deque = deque(maxlen=50)
        bar = tqdm(batcher(tokens, batch_size, ex_len, random_offset=True), total=steps_per_epoch)
        epoch_loss = 0
        epoch_start = step
        for batch, targets in bar:
            key, = jax.random.split(key, 1)
            t = t + 1
            opt_state, loss = update_fn(opt_state, t, batch, targets, key)
            loss_deque.append(loss)
            epoch_loss += loss
            if step % 50 == 0:
                avg_loss = np.mean([l.item() for l in loss_deque])
                bar.set_description(f'Epoch {epoch}, Step {step} - Average loss: {avg_loss:0.2e} (Last epoch loss: {last_epoch_loss:0.2e})')
            step += 1

        last_epoch_loss = epoch_loss.item() / (step - epoch_start)

        if epoch % save_period == 0:
            with (out_path / f'weights_{epoch:02d}.pkl').open('wb') as f:
                pickle.dump(get_params(opt_state), f)

    with (out_path / 'weights.pkl').open('wb') as f:
        pickle.dump(get_params(opt_state), f)