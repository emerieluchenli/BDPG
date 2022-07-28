from subproc_vec_env import *
from atari_wrappers import make_atari, wrap_deepmind
from model import Model
from train import Trainer


class run:
    def __init__(self, sess, args):
        self.args = args
        self.sess = sess
        self.model = Model(sess, optimizer_params={'learning_rate_actor': args['learning_rate_actor'],
                                                   'learning_rate_critic': args['learning_rate_critic'],
                                                   'epsilon': 1e-5}, args=args)
        self.trainer = Trainer(sess, self.model, args=args)

    def train(self):
        env = run.make_vec_env(env_id=self.args['env_name'], num_env=self.args['num_envs'], seed=self.args['env_seed'])
        env = VecNormalize(env, ob=False)
        # env.reset()

        print("\n\nBuilding the model...")
        self.model.build(env.observation_space.shape, env.action_space.n)
        print("Model is built successfully\n\n")

        print('Training...')
        try:
            self.trainer.train(env)
        except KeyboardInterrupt:
            print('keyboard interruption\n')
            env.close()

    def test(self, total_timesteps=1e5):
        env = run.make_vec_env(env_id=self.args['env_name'], num_env=1, seed=self.args['env_seed'])
        env = VecNormalize(env, ob=False)
        # env.reset()

        print("\n\nBuilding the model...")
        self.model.build(env.observation_space.shape, env.action_space.n)
        print("Model is built successfully\n\n")

        print('Testing...')
        try:
            self.trainer.test(total_timesteps=total_timesteps, env=env)
        except KeyboardInterrupt:
            print('keyboard interruption\n')
            env.close()

    @staticmethod
    def make_env(env_id, seed=None):
        env = make_atari(env_id)
        env.seed(seed)
        env = wrap_deepmind(env)
        return env

    @staticmethod
    def make_vec_env(env_id, num_env, seed, start_index=0):
        def make_thunk(rank):
            return lambda: run.make_env(
                env_id=env_id,
                seed=seed + rank)

        set_all_global_seeds(seed)
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
