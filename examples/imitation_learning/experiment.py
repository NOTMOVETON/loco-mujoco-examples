import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from experiment_launcher import run_experiment
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger

from imitation_lib.utils import BestAgentSaver

from loco_mujoco import LocoEnv
from utils import get_agent


def experiment(env_id: str = None,
               n_epochs: int = 5,
               n_steps_per_epoch: int = 100,
               n_steps_per_fit: int = 8196,
               n_eval_episodes: int = 100,
               n_epochs_save: int = 500,
               gamma: float = 0.99,
               results_dir: str = './logs',
               use_cuda: bool = True,
               seed: int = 83):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))

    # logging
    sw = SummaryWriter(log_dir=results_dir)     # tensorboard
    logger = Logger(results_dir=results_dir, log_name="logging", seed=seed, append=True)    # numpy

    print(f"Starting training {env_id}...")
    print(n_epochs)

    # create environment, agent and core
    mdp = LocoEnv.make(env_id)
    agent = get_agent(env_id, mdp, use_cuda, sw)
    core = Core(agent, mdp)
    
    for epoch in range(n_epochs):
        print(epoch)
        # train
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=False, render=False)

        # evaluate
        dataset = core.evaluate(n_episodes=n_eval_episodes)
        R_mean = np.mean(compute_J(dataset))
        J_mean = np.mean(compute_J(dataset, gamma=gamma))
        L = np.mean(compute_episodes_length(dataset))
        logger.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
        sw.add_scalar("Eval_R-stochastic", R_mean, epoch)
        sw.add_scalar("Eval_J-stochastic", J_mean, epoch)
        sw.add_scalar("Eval_L-stochastic", L, epoch)

        if epoch%3==0:
            print('saving agent', epoch)
            path = os.path.join(results_dir, f"agent_{epoch}_{R_mean}")
            agent.save(path, full_save=True)
            print('saving agent successed.')

    print("Finished.")


if __name__ == "__main__":
    run_experiment(experiment)
