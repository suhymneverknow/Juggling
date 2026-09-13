"""GPU-native PPO training for Mount1 using JAX, Flax, Optax and MJX."""

from __future__ import annotations
import json, pickle, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from envs.mount1_env import CONTROLLED_JOINTS, OBSERVATION_DIM, Mount1Env
from rewards.mount1_rewards import REWARD_VERSION


class ActorCritic(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, obs):
        def mlp(x, out):
            x=nn.tanh(nn.Dense(256)(x));x=nn.tanh(nn.Dense(256)(x));return nn.Dense(out)(x)
        mean=mlp(jnp.nan_to_num(obs),self.act_dim);value=mlp(jnp.nan_to_num(obs),1)[...,0]
        log_std=self.param("log_std",lambda k:jnp.full((self.act_dim,),-.6))
        return mean,jnp.clip(log_std,-4.,1.),value


def normal_log_prob(raw,mean,log_std):
    return jnp.sum(-.5*((raw-mean)/jnp.exp(log_std))**2-log_std-.5*jnp.log(2*jnp.pi),axis=-1)


def merge_stats(mean,var,count,x):
    bmean=jnp.mean(x,axis=0);bvar=jnp.var(x,axis=0);n=x.shape[0];delta=bmean-mean;total=count+n
    return mean+delta*n/total,(var*count+bvar*n+delta**2*count*n/total)/total,total


def normalize(obs,mean,var): return jnp.clip(jnp.nan_to_num((obs-mean)/jnp.sqrt(var+1e-8)),-10.,10.)


def init_wandb(args,num_updates):
    if not args.use_wandb:return None
    import wandb
    steps_per_update=args.num_envs*args.rollout_steps
    return wandb.init(project=args.wandb_project,name=args.wandb_run_name,mode=args.wandb_mode,dir=args.output_dir,config={**vars(args),"backend":"jax","reward_version":REWARD_VERSION,"observation_version":"mount1_relative_93_jax_v1","num_updates":num_updates,"steps_per_update":steps_per_update,"target_total_steps":num_updates*steps_per_update})


def train(args):
    print("JAX devices:",jax.devices())
    if args.device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()): raise RuntimeError("device=cuda requested but JAX has no GPU device")
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);args.output_dir=str(out)
    with (out/"run_config.json").open("w") as f:json.dump(vars(args),f,indent=2,default=str)
    env=Mount1Env(args);num_envs=args.num_envs;act_dim=len(CONTROLLED_JOINTS)
    reset_batch=jax.jit(jax.vmap(env.reset));step_batch=jax.jit(jax.vmap(env.step))
    key=jax.random.PRNGKey(args.seed);key,rk=jax.random.split(key);states=reset_batch(jax.random.split(rk,num_envs))
    model=ActorCritic(act_dim);key,pk=jax.random.split(key);params=model.init(pk,jnp.zeros((1,OBSERVATION_DIM)))
    tx=optax.chain(optax.clip_by_global_norm(args.max_grad_norm),optax.adam(args.lr));train_state=TrainState.create(apply_fn=model.apply,params=params,tx=tx)
    obs_mean=jnp.zeros((OBSERVATION_DIM,));obs_var=jnp.ones((OBSERVATION_DIM,));obs_count=jnp.array(1e-4);ret_rms_mean=jnp.array(0.);ret_rms_var=jnp.array(1.);ret_rms_count=jnp.array(1e-4);discounted=jnp.zeros((num_envs,))

    def collect(train_state,states,key,obs_mean,obs_var,discounted,rr_mean,rr_var,rr_count):
      def one(carry,_):
        states,key,discounted,rr_mean,rr_var,rr_count=carry;key,ak,rk=jax.random.split(key,3);obs_n=normalize(states.obs,obs_mean,obs_var);mean,log_std,value=train_state.apply_fn(train_state.params,obs_n);raw=mean+jnp.exp(log_std)*jax.random.normal(ak,mean.shape);action=jnp.tanh(raw);logp=normal_log_prob(raw,mean,log_std);stepped=step_batch(states,action);done=stepped.done;reward=stepped.reward
        discounted=.995*discounted+reward;rr_mean,rr_var,rr_count=merge_stats(rr_mean,rr_var,rr_count,discounted[:,None]);rr_mean,rr_var=rr_mean[0],rr_var[0];norm_reward=jnp.clip(reward/jnp.sqrt(rr_var+1e-8),-args.reward_clip,args.reward_clip);discounted=jnp.where(done,0.,discounted)
        resets=reset_batch(jax.random.split(rk,num_envs));sel=lambda a,b:jnp.where(done.reshape((num_envs,)+(1,)*(a.ndim-1)),b,a);data=jax.tree.map(sel,stepped.data,resets.data)
        next_states=stepped.replace(data=data,obs=sel(stepped.obs,resets.obs),step_count=jnp.where(done,0,stepped.step_count),success_hold_steps=jnp.where(done,0,stepped.success_hold_steps),board_lost_steps=jnp.where(done,0,stepped.board_lost_steps),last_action=sel(stepped.last_action,resets.last_action),previous_left_foot=sel(stepped.previous_left_foot,resets.previous_left_foot),previous_right_foot=sel(stepped.previous_right_foot,resets.previous_right_foot),kp=sel(stepped.kp,resets.kp),kd=sel(stepped.kd,resets.kd),rng=sel(stepped.rng,resets.rng),disturbance_force=sel(stepped.disturbance_force,resets.disturbance_force),disturbance_steps_remaining=jnp.where(done,resets.disturbance_steps_remaining,stepped.disturbance_steps_remaining),next_disturbance_step=jnp.where(done,resets.next_disturbance_step,stepped.next_disturbance_step))
        tr=(obs_n,action,logp,norm_reward,reward,done.astype(jnp.float32),value,stepped.metrics)
        return (next_states,key,discounted,rr_mean,rr_var,rr_count),tr
      return jax.lax.scan(one,(states,key,discounted,rr_mean,rr_var,rr_count),None,length=args.rollout_steps)
    collect_jit=jax.jit(collect)

    def update(train_state,key,obs,actions,old_logp,advantages,returns):
      n=obs.shape[0];perm=jax.random.permutation(key,n);usable=(n//args.minibatch_size)*args.minibatch_size;idx=perm[:usable].reshape((-1,args.minibatch_size))
      def epoch(carry,_):
       ts,key=carry;key,sk=jax.random.split(key);batches=jax.random.permutation(sk,idx,axis=0)
       def minibatch(ts,mb):
        def loss_fn(p):
         mean,ls,val=ts.apply_fn(p,obs[mb]);raw=jnp.arctanh(jnp.clip(actions[mb],-.999,.999));lp=normal_log_prob(raw,mean,ls);ratio=jnp.exp(lp-old_logp[mb]);adv=advantages[mb];pg=-jnp.mean(jnp.minimum(ratio*adv,jnp.clip(ratio,1-args.clip_coef,1+args.clip_coef)*adv));vl=.5*jnp.mean((val-returns[mb])**2);ent=jnp.sum(ls+.5*jnp.log(2*jnp.pi*jnp.e));return pg+args.vf_coef*vl-args.ent_coef*ent,(pg,vl,ent)
        (loss,aux),grads=jax.value_and_grad(loss_fn,has_aux=True)(ts.params);return ts.apply_gradients(grads=grads),(loss,*aux)
       ts,metrics=jax.lax.scan(minibatch,ts,batches);return (ts,key),metrics
      (train_state,key),metrics=jax.lax.scan(epoch,(train_state,key),None,length=args.epochs);return train_state,key,metrics
    update_jit=jax.jit(update)

    num_updates=args.max_iterations if args.max_iterations is not None else args.total_timesteps//(num_envs*args.rollout_steps);steps_per_update=num_envs*args.rollout_steps;target_total_steps=num_updates*steps_per_update
    print(f"steps_per_update={steps_per_update:,} target_total_steps={target_total_steps:,}")
    wandb_run=init_wandb(args,num_updates);start=time.time();steps=0
    for update_i in range(1,num_updates+1):
      flat=np.asarray(states.obs);obs_mean,obs_var,obs_count=merge_stats(obs_mean,obs_var,obs_count,jnp.asarray(flat))
      (states,key,discounted,ret_rms_mean,ret_rms_var,ret_rms_count),tr=collect_jit(train_state,states,key,obs_mean,obs_var,discounted,ret_rms_mean,ret_rms_var,ret_rms_count)
      obs,actions,logp,rews,raw_rews,dones,values,metrics=tr;_,_,next_value=train_state.apply_fn(train_state.params,normalize(states.obs,obs_mean,obs_var));adv=[];gae=jnp.zeros((num_envs,))
      for t in reversed(range(args.rollout_steps)):
       nv=next_value if t==args.rollout_steps-1 else values[t+1];gae=rews[t]+args.gamma*nv*(1-dones[t])-values[t]+args.gamma*args.gae_lambda*(1-dones[t])*gae;adv.append(gae)
      advantages=jnp.stack(adv[::-1]);returns=advantages+values;bobs=obs.reshape((-1,OBSERVATION_DIM));bact=actions.reshape((-1,act_dim));blogp=logp.reshape(-1);badv=advantages.reshape(-1);badv=(badv-jnp.mean(badv))/(jnp.std(badv)+1e-8);bret=returns.reshape(-1);key,uk=jax.random.split(key);train_state,key,losses=update_jit(train_state,uk,bobs,bact,blogp,badv,bret);steps+=num_envs*args.rollout_steps
      if update_i==1 or update_i%args.log_interval==0:
       host={k:float(np.asarray(jnp.mean(v))) for k,v in metrics.items()};loss_host=np.asarray(jax.device_get(losses));fps=int(steps/max(time.time()-start,1e-6));line=f"update={update_i:04d} steps={steps} reward={float(jnp.mean(raw_rews)):.3f} success={host['is_success']:.3f} fallen={host['robot_fallen']:.3f} fps={fps}";print(line,flush=True)
       if wandb_run:
        log={"train/update":update_i,"train/steps":steps,"train/target_steps":num_updates*num_envs*args.rollout_steps,"train/progress":update_i/num_updates,"train/fps":fps,"train/rollout_reward_mean":float(jnp.mean(raw_rews)),"train/rollout_reward_normalized_mean":float(jnp.mean(rews)),"loss/total":float(np.mean(loss_host[...,0])),"loss/policy":float(np.mean(loss_host[...,1])),"loss/value":float(np.mean(loss_host[...,2])),"loss/entropy":float(np.mean(loss_host[...,3])),"normalization/observation_running_count":float(obs_count),"normalization/observation_mean_abs":float(jnp.mean(jnp.abs(obs_mean))),"normalization/observation_std_mean":float(jnp.mean(jnp.sqrt(obs_var+1e-8))),"normalization/reward_return_std":float(jnp.sqrt(ret_rms_var+1e-8))};log.update({("reward/" if "reward" in k or "penalty" in k else "metric/")+k:v for k,v in host.items()});wandb_run.log(log,step=steps)
      if update_i%args.save_interval==0 or update_i==num_updates:
       payload={"params":jax.device_get(train_state.params),"obs_norm":{"mean":np.asarray(obs_mean),"var":np.asarray(obs_var),"count":float(obs_count)},"reward_norm":{"mean":float(ret_rms_mean),"var":float(ret_rms_var),"count":float(ret_rms_count)},"obs_dim":OBSERVATION_DIM,"act_dim":act_dim,"controlled_joints":CONTROLLED_JOINTS,"task":"mount1","backend":"jax","reward_version":REWARD_VERSION,"update":update_i,"steps":steps,"steps_per_update":steps_per_update,"target_total_steps":target_total_steps,"training_config":vars(args)}
       with (out/"latest.pkl").open("wb") as f:pickle.dump(payload,f)
    if wandb_run:wandb_run.finish()
    print("saved",out/"latest.pkl")
