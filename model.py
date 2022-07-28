import tensorflow as tf

from networks import AtariPolicy
from layers import a3c_entropy


class Model:
    def __init__(self, sess, optimizer_params, noise_amp=0., cliprange=0.1,
                 entropy_coeff=0.01, dim_z=2, max_gradient_norm=0.5,
                 args=None):
        self.train_batch_size = args['num_envs'] * args['unroll_time_steps']
        self.num_envs = args['num_envs']
        self.num_steps = args['unroll_time_steps']
        self.num_stack = args['num_stack']
        self.info_cap = args['info_cap']
        self.init_exp_coeff = args['init_exp_coeff']
        self.total_timesteps = args['total_timesteps']
        self.nepoch = args['nepoch']
        self.mini_bc = args['mini_batchsize']
        self.sess = sess
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_gradient_norm

        self.learning_rate_actor = optimizer_params['learning_rate_actor']
        self.learning_rate_critic = optimizer_params['learning_rate_critic']
        self.epsilon = optimizer_params['epsilon']
        self.policy = AtariPolicy

        self.dim_z = dim_z
        self.is_exploring = args['is_exploring']
        self.noise_amp = noise_amp
        self.cliprange = cliprange

    def init_input(self):
        self.X_input_train_shape = (
            None, self.img_height, self.img_width, self.num_classes * self.num_stack)
        self.X_input_step_shape = (
            None, self.img_height, self.img_width, self.num_classes * self.num_stack)

        self.actions = tf.placeholder(tf.int32, [None])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.R = tf.placeholder(tf.float32, [None]) # Bellman backup
        self.oldneglogpac = tf.placeholder(tf.float32, [None])

        self.learning_rate_ac = tf.placeholder(tf.float32, [])
        self.learning_rate_enc = tf.placeholder(tf.float32, [])
        self.learning_rate_d = tf.placeholder(tf.float32, [])

    def init_network(self):
        self.train_policy = self.policy(self.sess, self.X_input_train_shape, self.num_actions,
                                        reuse=False, keep_prob=0.9, dim_z=self.dim_z,
                                        is_training=True, noise_amp=0.)

        self.step_policy = self.policy(self.sess, self.X_input_step_shape, self.num_actions, reuse=True, keep_prob=1.,
                                       dim_z=self.dim_z, is_training=False,
                                       noise_amp=self.noise_amp, is_exploring=self.is_exploring)

        self.test_policy = self.policy(self.sess, self.X_input_step_shape, self.num_actions, reuse=True, keep_prob=1.,
                                       dim_z=self.dim_z, is_training=False, is_exploring=False, noise_amp=0.)

        # losses
        # policy
        negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.train_policy.policy_logits, labels=self.actions)

        ratio = tf.exp(self.oldneglogpac - negative_log_prob_action)
        pg_losses = - self.advantage * ratio
        pg_losses2 = - self.advantage * tf.clip_by_value(ratio, 1. - self.cliprange, 1. + self.cliprange)
        self.policy_gradient_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))  # PPO

        self.entropy_loss = - tf.reduce_mean(a3c_entropy(self.train_policy.policy_logits))

        # discriminator
        p_loss_true = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_policy.true_logits,
                                                    labels=tf.ones_like(self.train_policy.true_logits)))
        p_loss_gene = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_policy.gene_logits,
                                                    labels=tf.zeros_like(self.train_policy.gene_logits)))
        self.disc_loss = p_loss_true + p_loss_gene

        # auto encoder
        self.enc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_policy.true_logits,
                                                    labels=tf.zeros_like(self.train_policy.true_logits))) \
                       + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_policy.gene_logits,
                                                    labels=tf.ones_like(self.train_policy.gene_logits)))

        y_gene = tf.squeeze(self.train_policy.y_decoder_mean)

        reconstr_loss = tf.reduce_mean(tf.square(y_gene - self.R))
        self.ac_loss = self.policy_gradient_loss + 0.5 * reconstr_loss  + self.entropy_coeff * self.entropy_loss

        params = tf.compat.v1.trainable_variables()
        params_pr = [var for var in params if var.name.startswith("ret_z_prior")]
        params_d = [var for var in params if var.name.startswith("disc")]
        params_enc = [var for var in params if var.name.startswith("ret_encoder")]

        params_enc += params_pr
        params_ac = list(set(params) - set(params_d) - set(params_enc))

        # L2 regularization
        # L2_loss = tf.add_n([tf.compat.v1.nn.l2_loss(v) for v in params_ac if 'bias' not in v.name]) * 1e-4 #1e-3
        # self.ac_loss += L2_loss

        # apply gradients
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

        optimizer_ac = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_ac, epsilon=self.epsilon)
        grads_and_var_ac = optimizer_ac.compute_gradients(self.ac_loss, params_ac)
        grads_ac, var_ac = zip(*grads_and_var_ac)
        grads_ac, _grad_norm_ac = tf.clip_by_global_norm(grads_ac, self.max_grad_norm)
        grads_and_var_ac = list(zip(grads_ac, var_ac))
        self.optimize_ac = optimizer_ac.apply_gradients(grads_and_var_ac)
        self.optimize_ac = tf.group([self.optimize_ac, update_ops])

        optimizer_d = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_d, epsilon=self.epsilon)
        grads_and_var_d = optimizer_d.compute_gradients(self.disc_loss, params_d)
        grads_d, var_d = zip(*grads_and_var_d)
        grads_d, _grad_norm_d = tf.clip_by_global_norm(grads_d, self.max_grad_norm)
        grads_and_var_d = list(zip(grads_d, var_d))
        self.optimize_d = optimizer_d.apply_gradients(grads_and_var_d)
        self.optimize_d = tf.group([self.optimize_d, update_ops])

        optimizer_enc = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_enc, epsilon=self.epsilon)
        grads_and_var_enc = optimizer_enc.compute_gradients(self.enc_loss, params_enc)
        grads_enc, var_enc = zip(*grads_and_var_enc)
        grads_enc, _grad_norm_enc = tf.clip_by_global_norm(grads_enc, self.max_grad_norm)
        grads_and_var_enc = list(zip(grads_enc, var_enc))
        self.optimize_enc = optimizer_enc.apply_gradients(grads_and_var_enc)
        self.optimize_enc = tf.group([self.optimize_enc, update_ops])

    def build(self, observation_space_params, num_actions):
        self.img_height, self.img_width, self.num_classes = observation_space_params
        self.num_actions = num_actions
        self.init_input()
        self.init_network()


