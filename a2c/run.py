"""
    Build entry
"""
import tensorflow as tf

# import os

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import architecture as arch
import learn as model
import env


def run():
    config = tf.ConfigProto()

    # os.envrion['CUDA_VISIBLE_DEVICES'] = "0" # msg err
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        model.learn(policy=arch.A2CPolicy,
                    env=SubprocVecEnv([*env.make_train(all_=True)]),
                    n_steps=2048,
                    total_timesteps=1000000,
                    gamma=0.99,
                    lam=0.95,
                    lr=2e-4,
                    log_interval=10,
                    max_grad_norm=.5,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    )

        if __name__ == '__main__':
            run()
