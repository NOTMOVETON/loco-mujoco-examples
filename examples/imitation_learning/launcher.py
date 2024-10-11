from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = True

    N_SEEDS = 1

    launcher = Launcher(exp_name='loco_mujoco_evalution',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=1,  # only used for slurm
                        memory_per_core=1500,   # only used for slurm
                        n_exps_in_parallel=30,  # should not be used in slurm
                        days=0,     # only used for slurm
                        hours=0,    # only used for slurm
                        minutes=30,  # only used for slurm
                        use_timestamp=True,
                        )

    # default_params = dict(n_epochs=15,
    #                       n_steps_per_epoch=32768*8,
    #                       n_epochs_save=5,
    #                       n_eval_episodes=10,
    #                       n_steps_per_fit=10000,
    #                       use_cuda=USE_CUDA)

    default_params = dict(n_epochs=100,
                          n_steps_per_epoch=100000,
                          n_epochs_save=25,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA)

    env_ids = ["UnitreeA1.hard"]

    for env_id in env_ids:
        launcher.add_experiment(env_id__=env_id, **default_params)

    launcher.run(LOCAL, TEST)

