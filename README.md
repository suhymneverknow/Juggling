# Juggling

Mount1 is trained with the GPU-native JAX/MJX environment and PPO trainer.
The default configuration uses 4096 parallel environments with 128 rollout
steps per PPO update and logs to W&B.

## Train

```bash
./train_mount1.sh
```

Hydra overrides can be passed directly:

```bash
./train_mount1.sh num_envs=2048 max_iterations=1000 use_wandb=false
```

On the laboratory Slurm cluster:

```bash
ssh cluster_shias
cd ~/mengshuyu/juggling
/opt/gridview/slurm/bin/sbatch train_mount1.sh
```

Defaults are defined in `configs/mount1_train.yaml`. Run outputs are written to
`train/runs/mount1/YYYY-MM-DD/HH-MM-SS/`, and the latest checkpoint is
`latest.pkl`.

## Play a checkpoint

```bash
conda run --no-capture-output -n loco_mujoco \
  python scripts/play_mount1_policy.py \
  --checkpoint train/runs/mount1/<date>/<time>/latest.pkl
```

Add `--no-render` for a headless evaluation.

The simulator is MJX. MuJoCo is still used to load the XML model and display
checkpoints because MJX is MuJoCo's JAX backend.

## Balance task (preserved CPU implementation)

The independent balance task and its existing checkpoints remain available:

```bash
conda run --no-capture-output -n loco_mujoco python train/train_balance.py

conda run --no-capture-output -n loco_mujoco \
  python scripts/play_balance_policy.py \
  --checkpoint train/runs/balance/latest.pt

conda run --no-capture-output -n loco_mujoco python scripts/view_balance.py
```

Balance checkpoints are stored under `train/runs/balance/` and are not touched
by the Mount1 JAX/MJX cleanup.
