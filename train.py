import numpy as np
import tensorflow as tf
import pickle

from subproc_vec_env import TfRunningMeanStd, LearningRateDecay


class BaseTrainer:
    def __init__(self, sess, model, args):
        self.model = model
        self.args = args
        self.sess = sess

    def _load_model(self):
        # saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=2)

        # initialize variables
        self.init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        self.sess.run(self.init)

        # load model checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(self.args['checkpoint_dir'])

        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")

        else:
            print("No checkpoints available!\n\n")

    def init_model(self):
        # num of iterations
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

        # load model
        self._load_model()

    def save(self):
        print("Saving model...")
        self.saver.save(self.sess, self.args['checkpoint_dir'], self.global_step_tensor)
        print("Model saved")


class Trainer(BaseTrainer):
    def __init__(self, sess, model, args=None):
        super().__init__(sess, model, args)
        self.save_every = 240
        self.print_every = 80 # >= nlogg
        self.nlogg = 80

        self.sess = sess
        self.global_time_step = 0
        self.observation_stack = None
        self.dones = None
        self.env = None

        self.gamma = self.args['gamma']

        if self.model.is_exploring:
            self.ig_rms = TfRunningMeanStd(shape=(), scope='ig_rms', decay=0.8)
            self.model.init_exp_coeff /= np.sqrt(np.log(4) / 4)

    def train(self, env):
        self.init_model()
        self.env = env

        if self.model.is_exploring:
            self.ig_rms._set_mean_var_count()
        if self.env.ret_rms is not None:
            self.env.ret_rms._set_mean_var_count()
        if self.env.ob_rms is not None:
            self.env.ob_rms._set_mean_var_count()

        self.bc = int(self.model.num_steps * self.env.num_envs)
        self.model.mini_bc = min(self.model.mini_bc, self.bc)
        self.num_mini_bc = int(self.bc // self.model.mini_bc)
        self.num_iterations = int(self.model.total_timesteps // self.bc)

        self.observation_stack = np.zeros((self.env.num_envs,
                                           self.model.img_height,
                                           self.model.img_width,
                                           self.model.num_classes * self.model.num_stack),
                                          dtype=np.uint8)
        self.observation_stack = self.__observation_update(self.env.reset(), self.observation_stack)

        self.dones = [False for _ in range(self.env.num_envs)]

        self.sum_rew_cur_epi = np.zeros(self.env.num_envs)
        self.returns_all_env = []
        self.returns_all_env_cur_epoch = []

        self.global_step = self.global_step_tensor.eval(self.sess)

        num_grad_step = int((self.num_iterations - self.global_step) * self.model.nepoch * self.num_mini_bc)
        r_totrain = (self.num_iterations - self.global_step) / self.num_iterations
        self.learning_rate_actor_decayed = LearningRateDecay(v=self.model.learning_rate_actor * r_totrain,
                                                             nvalues=num_grad_step,
                                                             lr_decay_method='linear')
        self.learning_rate_critic_decayed = LearningRateDecay(v=self.model.learning_rate_critic * r_totrain,
                                                              nvalues=num_grad_step,
                                                              lr_decay_method='linear')

        self.train_input_shape = (self.model.train_batch_size, self.model.img_height, self.model.img_width,
                                  self.model.num_classes * self.model.num_stack)

        for self.local_step in range(self.global_step + 1, self.num_iterations + 1, 1):
            self.__rollout()

            inds = np.arange(len(self.mb_actions), dtype=int)
            for _ in range(self.model.nepoch):
                np.random.shuffle(inds)
                ind_sec = np.array_split(inds, self.num_mini_bc)

                for mbinds in ind_sec:
                    self.mb_obs_train = self.mb_obs.copy()[mbinds]
                    self.mb_actions_train = self.mb_actions.copy()[mbinds]
                    self.mb_neglogpac_train = self.mb_neglogpac.copy()[mbinds]
                    self.mb_R_train = self.mb_R.copy()[mbinds]
                    self.mb_advs_train = self.mb_advs.copy()[mbinds]

                    self.current_learning_rate_actor = self.learning_rate_actor_decayed.value()
                    self.current_learning_rate_critic = self.learning_rate_critic_decayed.value()

                    self.__rollout_update()

            if not self.local_step % self.nlogg:
                if len(self.returns_all_env_cur_epoch) > 0:
                    self.returns_all_env.append(np.mean(self.returns_all_env_cur_epoch))
                    self.returns_all_env_cur_epoch = []
                else:
                    self.returns_all_env.append(np.nan)

            self.global_step += 1
            self.global_step_assign_op.eval(session=self.sess, feed_dict={self.global_step_input: self.global_step})

            if not self.local_step % self.print_every:
                print(' - progress: ' + str(self.local_step / self.num_iterations * 100)[:5] + ' %')
                print('average score: ', self.returns_all_env[-1], '\n')

                if not self.local_step % self.save_every:
                    self.save()
                    with open('experiments/ret_atari.pkl', 'wb') as f:
                        pickle.dump(self.returns_all_env, f)

        self.save()
        with open('experiments/ret_atari.pkl', 'wb') as f:
            pickle.dump(self.returns_all_env, f)
        self.env.close()

    def test(self, total_timesteps, env):
        self.init_model()

        if self.ig_rms is not None:
            self.ig_rms._set_mean_var_count()
        if env.ret_rms is not None:
            env.ret_rms._set_mean_var_count()
        if env.ob_rms is not None:
            env.ob_rms._set_mean_var_count()

        observation_stack = np.zeros((1,
                                      self.model.img_height,
                                      self.model.img_width,
                                      self.model.num_classes * self.model.num_stack),
                                     dtype=np.uint8)
        observation_stack = self.__observation_update(env.reset(), observation_stack)
        dones = [False]

        for _ in range(total_timesteps):
            actions = self.model.test_policy.get_action(observation_stack, dones)
            observation, rewards0, rewards, dones, info = env.step(actions)

            env.render()

            if dones[0]:
                observation_stack[0] *= 0
            observation_stack = self.__observation_update(observation, observation_stack)
        env.close()

    def __rollout(self):

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpac = [], [], [], [], [], []

        for t in range(self.model.num_steps):
            # choose action
            actions, values, neglogpac = \
                self.model.step_policy.get_action_value_neglogpac(self.observation_stack, self.dones)

            mb_obs.append(self.observation_stack.copy())
            mb_actions.append(actions.copy())
            mb_values.append(values.copy())
            mb_dones.append(self.dones.copy())
            mb_neglogpac.append(neglogpac.copy())

            # take action
            observation, rewards0, rewards, dones, info = self.env.step(actions)

            self.dones = dones
            for n, (done, rew) in enumerate(zip(dones, rewards0)):
                self.sum_rew_cur_epi[n] += rew
                if done:
                    self.observation_stack[n] *= 0
                    self.returns_all_env_cur_epoch.append(self.sum_rew_cur_epi[n])
                    self.sum_rew_cur_epi[n] = 0.

            # update stacked observation
            self.observation_stack = self.__observation_update(observation, self.observation_stack)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape(np.shape(mb_rewards))
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_neglogpac = np.asarray(mb_neglogpac, dtype=np.float32)

        last_values = self.model.step_policy.get_value(self.observation_stack, self.dones)

        # bootstrap rewards
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.model.num_steps)):
            if t == self.model.num_steps - 1:
                nextnonterminal = 1. - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1. - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * 0.95 * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        if self.model.is_exploring:
            # info gain for exploration
            curr_exp_coeff = self._exp_coeff()

            info_gain = self.model.step_policy.get_info_gain(mb_obs.reshape(self.train_input_shape), mb_returns.flatten())
            if self.model.info_cap is not None:
                info_gain = np.minimum(info_gain, self.model.info_cap)

            self.ig_rms.update(info_gain)
            info_gain = (info_gain - self.ig_rms.mean) / np.sqrt(self.ig_rms.var + 1e-8)
            info_gain = info_gain.reshape((self.model.num_steps, self.env.num_envs))
            info_gain = curr_exp_coeff * info_gain / self.gamma

            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.model.num_steps)):
                if t == self.model.num_steps - 1:
                    nextnonterminal = 1. - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1. - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                nextvalues += info_gain[t]
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * 0.95 * nextnonterminal * lastgaelam
            # DO NOT update R with info_gain

        # conversion from (time_steps, num_envs) to (num_envs, time_steps)
        self.mb_obs = mb_obs.swapaxes(1, 0).reshape(self.train_input_shape)
        self.mb_actions = mb_actions.swapaxes(1, 0).flatten()
        self.mb_neglogpac = mb_neglogpac.swapaxes(1, 0).flatten()
        self.mb_R = mb_returns.swapaxes(1, 0).flatten()
        self.mb_advs = mb_advs.swapaxes(1, 0).flatten()

    def __rollout_update(self):
        # # Normalize the advantages
        # self.mb_advs_train = (self.mb_advs_train - self.mb_advs_train.mean()) / (self.mb_advs_train.std() + 1e-8)

        loss_d, _ = self.sess.run([self.model.disc_loss, self.model.optimize_d],
                                  feed_dict={self.model.train_policy.X_input: self.mb_obs_train,
                                             self.model.train_policy.y: self.mb_R_train,
                                             self.model.learning_rate_d: self.current_learning_rate_critic}) #2.5e-4

        loss_enc, _ = self.sess.run([self.model.enc_loss, self.model.optimize_enc],
                                    feed_dict={self.model.train_policy.X_input: self.mb_obs_train,
                                               self.model.train_policy.y: self.mb_R_train,
                                               self.model.R: self.mb_R_train,
                                               self.model.learning_rate_enc: self.current_learning_rate_critic})

        loss_ac, _ = self.sess.run([self.model.ac_loss, self.model.optimize_ac],
                                   feed_dict={self.model.train_policy.X_input: self.mb_obs_train,
                                              self.model.train_policy.y: self.mb_R_train,
                                              self.model.R: self.mb_R_train,
                                              self.model.actions: self.mb_actions_train,
                                              self.model.advantage: self.mb_advs_train,
                                              self.model.oldneglogpac: self.mb_neglogpac_train,
                                              self.model.learning_rate_ac: self.current_learning_rate_actor})

    def __observation_update(self, new_observation, old_observation_stack):
        updated_observation = np.roll(old_observation_stack, shift=-1, axis=3)
        updated_observation[:, :, :, -1] = new_observation[:, :, :, 0]
        return updated_observation

    def _safemean(self, x):
        if len(x) == 0:
            return np.nan
        else:
            return np.mean(x)

    def _exp_coeff(self):
        t = np.maximum(4, self.global_step * self.model.nepoch)# * self.num_mini_bc)
        return self.model.init_exp_coeff * np.sqrt(np.log(t) / t)


