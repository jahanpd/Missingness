import jax
import jax.numpy as jnp

def attention_lr(d_model, warmup_steps):
  def schedule(step_num):
    return ((d_model)**-0.5) * jnp.minimum(1e-4 / (d_model**(-0.5)), step_num * (warmup_steps**-1.5))
  return schedule
