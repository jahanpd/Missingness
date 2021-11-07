import jax
import jax.numpy as jnp

def attention_lr(d_model, warmup_steps, final_lr=8e-4):
  def schedule(step_num):
    return ((d_model)**-0.5) * jnp.minimum(final_lr / (d_model**(-0.5)), step_num * (warmup_steps**-1.5))
  return schedule

def linear_increase(nsteps, minlr, maxlr):
  def schedule(step_num):
    slope = (maxlr - minlr) / nsteps
    return slope * step_num + minlr
  return schedule