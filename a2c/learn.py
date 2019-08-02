import numpy as np

from baselines import logger

import time
import os

from .a2_c import Model
from .runner import Runner
from .utils import explained_variance


def play(policy, env):
    """
        Plays an env using the trained model
    """
    obs_space = env.observation_space
    action_space = env.actions_space

    model = Model(policy=policy,
                  obsv_space=obs_space,
                  action_space=action_space,
                  n_steps=n_steps,
                  n_envs=n_envs,
                  vf_coeff=vf_coeff,
                  ent_coeff=ent_coeff,
                  max_grad_norm=max_grad_norm)


def learn(policy, env, n_steps, total_timesteps, gamma, lam, **kwargs):
    """
        Instantiates the step model, the train model
        and the runner object.

        - Trains in two phases:
            Fetches experiences batch
            Trains batch
    """
    vf_coeff = kwargs.get('vf_coeff')
    ent_coeff = kwargs.get('ent_coeff')
    lr = kwargs.get('lr')
    max_grad_norm = kwargs.get('max_grad_norm')
    log_interval = kwargs.get('log_interval')

    n_optepochs = 4
    n_mini_batches = 8

    n_envs = env.num_envs
    obsv_space = env.observation_space
    act_space = env.action_space

    batch_size = n_steps * n_envs
    batch_train_size = batch_size // n_mini_batches

    model = Model(policy=policy,
                  obsv_space=obsv_space,
                  action_space=act_space,
                  n_steps=n_steps,
                  n_envs=n_envs,
                  vf_coef=vf_coeff,
                  ent_coef=ent_coeff,
                  max_grad_norm=max_grad_norm)
    load_path = '.model/a2c/model.ckpt'
    model.load(load_path)

    runner = Runner(env=env,
                    model=model,
                    n_steps=n_steps,
                    total_timesteps=total_timesteps,
                    gamma=gamma,
                    lam=lam)

    # Start total timer
    tfirst_start = time.time()

    for update in range(1, total_timesteps // batch_size + 1):
        t_start = time.time()

        # mini batch
        obs, actions, returns, values = runner.run()

        # For each mini batch, find loss
        mb_losses = []

        indices = np.arange(batch_size)

        for _ in range(n_optepochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, batch_train_size):
                end = start + batch_train_size
                mb_indxs = indices[start:end]
                slices = [arr[mb_indxs]
                          for arr in (obs, actions, returns, values)
                          ]
                mb_losses.append(model.train(*slices, l_r=lr))

        # Feedforward: get losses and update
        losses_val = np.mean(mb_losses, axis=0)
        t_end = time.time()

        # Frames per Second
        fps = batch_size // (t_end - t_start)

        if not update % log_interval or update == 1:
            """
                Computes fraction of variance of y_pred about y

                ev = 1 - var[y - y_pred] / var[y]

                ev=0: might as well have predicted zero
                ev<0: worse that just predicting zero
                ev=1: perfect prediction # Goal -> ev close to one
            """
            e_v = explained_variance(values, returns)
            logger.record_tabular('n_updates', update)
            logger.record_tabular('fps', fps)
            logger.record_tabular('total_timesteps', update * batch_size)
            logger.record_tabular('policy_loss', float(losses_val[0]))
            logger.record_tabular('value_loss', float(losses_val[1]))
            logger.record_tabular('policy_entropy', float(losses_val[2]))
            logger.record_tabular('explained_variance', float(e_v))
            logger.record_tabular('elapsed_time', float(t_end - tfirst_start))
            logger.dump_tabular()

            save_path = os.path.join('.models/', str(update), 'model.ckpt')
            print(f'saving to {save_path}')
            model.save(save_path)

    env.close()
