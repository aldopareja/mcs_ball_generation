from dataclasses import dataclass, replace
import pickle

import h5py
import numpy as onp

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import jax
from jax import jit, vmap
from jax import numpy as jnp
from jax.random import PRNGKey, split

import optax
from optax import sigmoid_binary_cross_entropy

from flax_transformer import TransformerConfig, TransformerStack

def get_infinite_batch_indices(batch_size, data_size):
  i = 0
  e = 0
  it = 0
  indices = onp.arange(data_size)
  indices = onp.random.permutation(indices)
  while True:
    if i + batch_size > len(indices):
      print('epoch_done', e)
      indices = onp.random.permutation(indices)
      i = 0
      e += 1
    yield indices[i:i+batch_size], it
    i += batch_size
    it += 1

def get_test_batch_indices(batch_size, data_size):
  i = 0
  it = 0
  indices = onp.arange(data_size)
  while True:
    if i + batch_size >= len(indices):
      yield indices[i:]
      break
    yield indices[i:i+batch_size]
    i += batch_size
    it += 1

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def load_data(data_path):
  with h5py.File(data_path) as f:
    _, rgbs, depths, labels = map(lambda x: f[x][:], ['masks', 'rgb_clouds', 'depth_clouds', 'labels'])
  rgbs = rgbs.astype(onp.uint8).astype(onp.float32)
  assert rgbs.shape[1] == 3 and depths.shape[1] == 3 and len(labels.shape) == 1
  colored_point_cloud = onp.zeros((rgbs.shape[0],6,MAX_SIZE),dtype=onp.float32)
  #concatenate the point clouds
  colored_point_cloud[:,onp.arange(3)] = rgbs
  colored_point_cloud[:,onp.arange(3)+3] = depths
  
  #swapaxes transformer needs B x SeqLenght x num_vars
  colored_point_cloud = onp.swapaxes(colored_point_cloud, 2, 1)
  return colored_point_cloud, labels[:,None] #add second dimension for broadcastability
        
@dataclass
class OptimCfg:
    max_lr: float = 1e-3
    num_steps: int = 10000
    pct_start: float = 0.01
    div_factor: float = 1e1
    final_div_factor: float = 1e1
    weight_decay: float = 0.00005
    gradient_clipping: float = 5.0
  
def load_chkpt(load_idx, chkpt_folder):
    with open(f"{chkpt_folder}params_{load_idx}", "rb") as f:
        params = pickle.load(f)
    with open(f"{chkpt_folder}opt_state_{load_idx}", "rb") as f:
        opt_state = pickle.load(f)
    with open(f"{chkpt_folder}key_{load_idx}", "rb") as f:
        loaded_key = pickle.load(f)
    return params, opt_state, loaded_key


def save_chkpt(save_idx, chkpt_folder, params, opt_state, old_key, extra=""):
    with open(f"{chkpt_folder}params_{save_idx}_{extra}", "wb") as f:
        pickle.dump(params, f)
    with open(f"{chkpt_folder}opt_state_{save_idx}_{extra}", "wb") as f:
        pickle.dump(opt_state, f)
    with open(f"{chkpt_folder}key_{save_idx}_{extra}", "wb") as f:
        pickle.dump(old_key, f)
        
def initialize_model(key,
                     obs_length,
                     num_input_vars,
                     t_cfg):
  
  _, *sks = split(key, 3)
  m = TransformerStack(config=t_cfg)
  
  variables = m.init(
      {"params": sks[0], "dropout": sks[1],},
      jnp.ones((2, obs_length, num_input_vars)),
  )
  state, params = variables.pop("params")
  del variables
  return m, state, params
        
def initialize_model_and_state(
    key: PRNGKey,
    obs_length,
    num_input_vars,
    optim_cfg: OptimCfg,
    t_cfg: TransformerConfig,
    load_idx=None,
    chkpt_folder=None,
):
  
  key, *sks = split(key, 10)
  
  m, state, params = initialize_model(sks[0], obs_length, num_input_vars, t_cfg)
  
  
  schedule = optax.cosine_onecycle_schedule(
      optim_cfg.num_steps,
      optim_cfg.max_lr,
      optim_cfg.pct_start,
      optim_cfg.div_factor,
      optim_cfg.final_div_factor,
  )
  tx = optax.chain(
      optax.clip_by_global_norm(optim_cfg.gradient_clipping),
      optax.adamw(learning_rate=schedule, weight_decay=optim_cfg.weight_decay),
  )
  # tx = optax.sgd(learning_rate=1e-9)
  opt_state = tx.init(params)
  
  if load_idx is not None:
    params, _, _ = load_chkpt(load_idx, chkpt_folder)
    opt_state = tx.init(params)
  
  return m, tx, opt_state, params, state


def update_step(
  apply_func,
  point_cloud,
  labels,
  opt_state,
  params,
  state,
  key,
):
  _, dropout_key = split(key)
  
  def loss(params):
    y_hat = apply_func(
      {'params': params, **state},
      point_cloud,
      rngs = {"dropout": dropout_key}
    )
    
    l = sigmoid_binary_cross_entropy(y_hat, labels)
    assert l.shape[1] == 1, len(l.shape) == 2
    
    return l.mean()
  
  l, grads = jax.value_and_grad(loss)(params)
  
  updates, opt_state = tx.update(
    grads,
    opt_state,
    params,
  )
  params = optax.apply_updates(params, updates)
  norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(params)[0])
  return opt_state, params, l, norm

@jit
def binary_auc(y_hat, labels):
  assert len(labels.shape) == 2 and labels.shape[1] == 1
  
  labels = labels.reshape(-1)
  y_hat = y_hat.reshape(-1)
  
  pos = jnp.where(labels == 1, y_hat, jnp.nan)
  neg = jnp.where(labels == 0, y_hat, jnp.nan)
  auc_func = vmap(vmap(lambda x,y: (x>y+1e-7) & (jnp.logical_not(jnp.isnan(x))) & (jnp.logical_not(jnp.isnan(y))), in_axes=(0,None)),in_axes=(None,0))
  
  auc = auc_func(pos,neg)
  return auc.sum()/(labels.sum() * (1 - labels).sum())
  

def evaluate_model(apply_func, params ,point_cloud_test, labels_test, batch_size):
  eval_yielder = get_test_batch_indices(batch_size,labels_test.shape[0])
  all_yhat = []
  for idx in eval_yielder:
    y_hat = apply_func({'params': params},
                       point_cloud_test[idx])
    all_yhat.append(jax.lax.stop_gradient(y_hat))
  all_yhat = jnp.concatenate(all_yhat)
  return binary_auc(all_yhat, labels_test)
    
        
if __name__ == "__main__":
  import wandb

  MAX_SIZE = 2000
  
  wandb.login()
  wandb.init(project="ball_detector")
  
  cfg = AttrDict()
  
  cfg.key = PRNGKey(639265)
  
  ## checkpoint params
  cfg.save_params = 10
  cfg.print_every = 1
  cfg.chkpt_folder = "chkpts/"
  cfg.load_idx = 1
  cfg.eval_every = 100
  save_idx = 0 if cfg.load_idx is None else cfg.load_idx + 1


  ## optim config
  cfg.optim_cfg = OptimCfg(
      max_lr=1e-3,
      num_steps=int(100000),
      pct_start=0.01,
      div_factor=1e2,
      final_div_factor=1e0,
      weight_decay=0.0005,
      gradient_clipping=5.0,
  )
  
  ## transformer config
  cfg.t_cfg = TransformerConfig(
    num_heads= 4,
    num_enc_layers=1,
    num_dec_layers=1,
    dropout_rate=0.1,
    deterministic=False,
    d_model=32, #should be a multiple of the number of heads.
    add_positional_encoding=False,
    obs_emb_hidden_sizes=(120,),
    num_latents=1
  )

  
  m, tx, opt_state, params, state = initialize_model_and_state(
        key=PRNGKey(1574),
        obs_length=MAX_SIZE,
        optim_cfg=cfg.optim_cfg,
        t_cfg=cfg.t_cfg,
        num_input_vars=6,
        chkpt_folder=cfg.chkpt_folder,
        load_idx=cfg.load_idx,
    )
  
  eval_cfg = replace(cfg.t_cfg,**{'deterministic': True})
  eval_m, _, _ = initialize_model(key=PRNGKey(6734),obs_length=MAX_SIZE,num_input_vars=6,t_cfg=eval_cfg)

  
  # data params
  cfg.data_path = 'soccer_balls_data_final_2.h5'
  cfg.batch_size = 48
  cfg.test_batch_size = 100
  cfg.test_size = 20000
  point_cloud, labels = load_data(cfg.data_path)
  
  wandb.config.update(cfg, allow_val_change=True)
  
  
  pc_train, pc_test, labels_train, labels_test = train_test_split(point_cloud, labels, test_size = cfg.test_size)
  train_data_size = len(labels_train)
  
  train_batch_yielder = get_infinite_batch_indices(cfg.batch_size, train_data_size)
  
  key = cfg.key
  while True:

    if cfg.load_idx:
        _, _, loaded_key = load_chkpt(cfg.load_idx, cfg.chkpt_folder)
    else:
        loaded_key = cfg.key
    
    idx, it = next(train_batch_yielder)
    step_pc, step_labels = jnp.array(pc_train[idx]), jnp.array(labels_train[idx])
    
    # opt_state, params, l, norm = update_step(
    opt_state, params, l, norm = jit(update_step,static_argnums=(0,))(
      m.apply,
      step_pc,
      step_labels,
      opt_state,
      params,
      state,
      key
    )
    
    if it % cfg.print_every == 0:
      print(it, l)
    wandb.log({"iteration": it, "loss": l})
      
    if jnp.isnan(l) or l == 0 or jnp.isinf(l):
      print("failed", it, l)
      break
    
    if (it+1) % cfg.save_params == 0:
      save_chkpt(save_idx, cfg.chkpt_folder, params, opt_state, key)
    
    if (it+1) % cfg.eval_every == 0:
      # auc = evaluate_model(
      auc = jit(evaluate_model,static_argnums=(0,4))(
        eval_m.apply, 
        params, 
        pc_test, 
        labels_test, 
        cfg.test_batch_size
      )
      print("auc", auc)
      wandb.log({"iteration": it, "auc": auc.item()})
      
    key, _ = split(key)
    
    