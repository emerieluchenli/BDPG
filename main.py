import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from numpy import random

from run import run


def main(args={'checkpoint_dir': "chp/",
         'num_envs': 16,
         'env_name': "AsterixNoFrameskip-v4",
         'env_seed': random.randint(0, 10000),
         'unroll_time_steps': 128,
         'num_stack': 4,
         'info_cap': 1e5,
         'init_exp_coeff': 1e-3,
         'total_timesteps': 16e7,
         'learning_rate_actor': 8e-4,
         'learning_rate_critic': 9e-4,
         'nepoch': 4,
         'mini_batchsize': 256,
         'gamma': 0.99,
         'is_exploring': True},
         is_train=True):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=args['num_envs'],
                            inter_op_parallelism_threads=args['num_envs'])
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with sess:
        runner = run(sess, args)

        if is_train:
            runner.train()
        else:
            runner.test(total_timesteps=int(1e7))


if __name__ == '__main__':
    main()

# main()
