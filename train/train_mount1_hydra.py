"""Hydra entry point for JAX/MJX Mount1 PPO."""
import argparse, sys
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train.train_mount1 import train

@hydra.main(version_base="1.3",config_path="../configs",config_name="mount1_train")
def main(cfg:DictConfig):
    values=OmegaConf.to_container(cfg,resolve=True);values["output_dir"]=str(Path(HydraConfig.get().runtime.output_dir).resolve());train(argparse.Namespace(**values))
if __name__=="__main__":main()
