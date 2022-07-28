import numpy as np
import tensorflow as tf

from layers import fully_connected, flatten, orthogonal_initializer, noise_and_argmax, conv2d


class AtariPolicy:
    def __init__(self, sess, input_shape, num_actions, reuse=True, keep_prob=1., noise_amp=0.,
                 dim_z=8, is_training=True, is_exploring=False):
        self.sess = sess
        self.input_shape = input_shape
        self.reuse = reuse
        self.keep_prob = keep_prob
        self.dim_z = dim_z
        self.activation = tf.nn.tanh
        self.init_scale = 1.
        self.dim_realval = 32
        self.batch_norm = True

        with tf.name_scope('policy_input'):
            self.X_input = tf.placeholder(tf.uint8, input_shape)

        if is_training or is_exploring:
            with tf.name_scope('policy_input'):
                self.y = tf.placeholder(tf.float32, [None])

        with tf.variable_scope("common", reuse=reuse):
            conv1 = conv2d('conv1', tf.cast(self.X_input, tf.float32) / 255., num_filters=32, kernel_size=(8, 8),
                           padding='VALID', stride=(4, 4),
                           initializer=orthogonal_initializer(np.sqrt(2.)), activation=tf.nn.relu,
                           is_training=is_training, batchnorm_enabled=self.batch_norm)

            conv2 = conv2d('conv2', conv1, num_filters=64, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                           initializer=orthogonal_initializer(np.sqrt(2.)), activation=tf.nn.relu,
                           is_training=is_training, batchnorm_enabled=self.batch_norm)

            conv3 = conv2d('conv3', conv2, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2.)), activation=tf.nn.relu,
                           is_training=is_training, batchnorm_enabled=self.batch_norm)

            self.state_emb = fully_connected('fc4', flatten(conv3), output_dim=512,
                                             initializer=orthogonal_initializer(1.),
                                             activation=tf.nn.tanh, dropout_keep_prob=keep_prob)

        with tf.variable_scope("policy", reuse=reuse):
            p_fc1 = fully_connected('policy_fc1', self.state_emb, output_dim=256, activation=self.activation,
                                   initializer=orthogonal_initializer(self.init_scale), dropout_keep_prob=keep_prob)

            self.policy_logits = fully_connected('policy_logits', p_fc1, output_dim=num_actions,
                                                 initializer=orthogonal_initializer(1.))

        value, z_sample_prior = self._sample_x(return_z=True)

        with tf.name_scope('state_value'):
            self.value_s = tf.squeeze(value)

        with tf.name_scope('action'):
            if noise_amp > 0:
                self.action_s = noise_and_argmax(self.policy_logits, noise_amp)
            else:
                self.action_s = tf.argmax(self.policy_logits, 1)

            self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.policy_logits, labels=self.action_s)

        if is_training or is_exploring:
            self.y_encoder_mean, self.y_encoder_stddev = self._Q_encode(self.y, self.state_emb, reuse=reuse)

            self.yz_sample = self.y_encoder_mean + \
                             self.y_encoder_stddev * tf.random_normal(tf.shape(self.y_encoder_mean), 0, 1,
                                                                      dtype=tf.float32)

            self.y_decoder_mean = self._P_decode(self.yz_sample, self.state_emb, reuse=True)

            # discriminate
            self.true_logits = self._discriminator(value, z_sample_prior, self.state_emb, reuse=reuse)
            self.gene_logits = self._discriminator(self.y, self.yz_sample, self.state_emb, reuse=True)

        if is_exploring:
            self.g_encoder_mean, self.g_encoder_stddev = self._Q_encode(value, self.state_emb, reuse=True)

            p = tf.distributions.Normal(self.y_encoder_mean, self.y_encoder_stddev)
            q = tf.distributions.Normal(self.g_encoder_mean, self.g_encoder_stddev)
            self.info_gain = tf.reduce_mean(p.kl_divergence(q), -1)

    def _prior(self, c, reuse=True):
        with tf.variable_scope("ret_z_prior", reuse=reuse):
            fc1 = fully_connected('fc1', c, output_dim=128,
                                  initializer=orthogonal_initializer(self.init_scale),
                                  activation=self.activation, dropout_keep_prob=self.keep_prob)

            prior_mean = fully_connected('mean', fc1, output_dim=self.dim_z, initializer=orthogonal_initializer(1.),
                                         activation=tf.nn.tanh)

        return prior_mean, tf.ones_like(prior_mean)

    def _Q_encode(self, x, c, reuse=True):
        with tf.variable_scope("ret_encoder", reuse=reuse):
            fc_x = fully_connected('fc_x', tf.reshape(x, (-1, 1)), output_dim=self.dim_realval, activation=tf.nn.tanh,
                                   initializer=orthogonal_initializer(1.),
                                   dropout_keep_prob=self.keep_prob)

            fc1 = fully_connected('fc1', tf.concat(axis=1, values=[fc_x, c]), output_dim=256,
                                  initializer=orthogonal_initializer(self.init_scale),
                                  activation=self.activation, dropout_keep_prob=self.keep_prob)

            fc2 = fully_connected('fc2', fc1, output_dim=128, activation=self.activation,
                                  initializer=orthogonal_initializer(self.init_scale), dropout_keep_prob=self.keep_prob)

            encoder_mean = fully_connected('mean', fc2, output_dim=self.dim_z,
                                           initializer=orthogonal_initializer(1.))

            encoder_stddev = fully_connected('stddev', fc2, output_dim=self.dim_z,
                                             initializer=tf.zeros_initializer())

        return encoder_mean, tf.nn.softplus(encoder_stddev)

    def _P_decode(self, z, c, reuse=True):
        with tf.variable_scope("ret_decoder", reuse=reuse):
            fc_z = fully_connected('fc_z', z, output_dim=self.dim_realval, activation=tf.nn.tanh,
                                   initializer=orthogonal_initializer(1.),
                                   dropout_keep_prob=self.keep_prob)

            fc1 = fully_connected('fc1', tf.concat(axis=1, values=[fc_z, c]), output_dim=256,
                                  initializer=orthogonal_initializer(self.init_scale),
                                  activation=self.activation, dropout_keep_prob=self.keep_prob)

            fc2 = fully_connected('fc2', fc1, output_dim=128, activation=self.activation,
                                  initializer=orthogonal_initializer(self.init_scale), dropout_keep_prob=self.keep_prob)

            x_decoder_mean = fully_connected('mean', fc2, output_dim=1, initializer=orthogonal_initializer(1.))

        return x_decoder_mean

    def _discriminator(self, x, z, c, reuse=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            fc_x = fully_connected('fc_x', tf.reshape(x, (-1, 1)), output_dim=self.dim_realval,
                                   initializer=orthogonal_initializer(1.),
                                   dropout_keep_prob=self.keep_prob, activation=tf.nn.tanh)

            fc_z = fully_connected('fc_z', z, output_dim=self.dim_realval, activation=tf.nn.tanh,
                                   initializer=orthogonal_initializer(1.),
                                   dropout_keep_prob=self.keep_prob)

            fc1 = fully_connected('fc1', tf.concat(axis=1, values=[fc_x, fc_z, c]), output_dim=256,
                                  initializer=orthogonal_initializer(np.sqrt(2.)),
                                  activation=tf.nn.relu, dropout_keep_prob=self.keep_prob)

            fc2 = fully_connected('fc2', fc1, output_dim=128, activation=tf.nn.leaky_relu,
                                  dropout_keep_prob=self.keep_prob,
                                  initializer=orthogonal_initializer(np.sqrt(2/(1+0.2**2)))) #leaky alpha = 0.2

            logits = fully_connected('logits', fc2, output_dim=1, initializer=orthogonal_initializer(1.))

        return logits

    def _sample_x(self, return_z=False):
        mean, stddev = self._prior(self.state_emb, reuse=self.reuse)
        z_sample = mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
        x_mean = self._P_decode(z_sample, self.state_emb, reuse=self.reuse)
        if return_z:
            return x_mean, z_sample
        else:
            return x_mean

    def get_value(self, observation, *_args, **_kwargs):
        return self.sess.run(self.value_s, {self.X_input: observation})

    def get_action(self, observation, *_args, **_kwargs):
        return self.sess.run(self.action_s, feed_dict={self.X_input: observation})

    def get_action_value(self, observation, *_args, **_kwargs):
        return self.sess.run([self.action_s, self.value_s], feed_dict={self.X_input: observation})

    def get_action_value_neglogpac(self, observation, *_args, **_kwargs):
        return self.sess.run([self.action_s, self.value_s, self.neglogpac], feed_dict={self.X_input: observation})

    def get_info_gain(self, observation, R, *_args, **_kwargs):
        return self.sess.run(self.info_gain, {self.X_input: observation, self.y: R})