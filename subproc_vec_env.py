import numpy as np
import contextlib
import os
import random
from abc import ABC, abstractmethod
import multiprocessing as mp
import tensorflow as tf


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        # for remote in self.work_remotes:
        #     remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count, decay=1.):

    delta = batch_mean - mean
    tot_count = count + batch_count

    count = int(count * decay)
    batch_count = tot_count - count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def get_session():
    return tf.get_default_session()


class TfRunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), scope='', decay=1.):
        sess = get_session()

        self._new_mean = tf.placeholder(shape=shape, dtype=tf.float64)
        self._new_var = tf.placeholder(shape=shape, dtype=tf.float64)
        self._new_count = tf.placeholder(shape=(), dtype=tf.float64)
        self.decay = decay


        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self._mean  = tf.get_variable('mean',  initializer=np.zeros(shape, 'float64'),      dtype=tf.float64, trainable=False)
            self._var   = tf.get_variable('std',   initializer=np.ones(shape, 'float64'),       dtype=tf.float64, trainable=False)
            self._count = tf.get_variable('count', initializer=np.full((), epsilon, 'float64'), dtype=tf.float64, trainable=False)

        self.update_ops = tf.group([
            self._var.assign(self._new_var),
            self._mean.assign(self._new_mean),
            self._count.assign(self._new_count)
        ])

        sess.run(tf.variables_initializer([self._mean, self._var, self._count]))
        self.sess = sess
        # self._set_mean_var_count()

    def _set_mean_var_count(self):
        self.mean, self.var, self.count = self.sess.run([self._mean, self._var, self._count])

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_mean, new_var, new_count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count, self.decay)
        
        self.sess.run(self.update_ops, feed_dict={
            self._new_mean: new_mean,
            self._new_var: new_var,
            self._new_count: new_count
        })

        self._set_mean_var_count()


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = TfRunningMeanStd(shape=self.observation_space.shape, scope='ob_rms') if ob else None
        self.ret_rms = TfRunningMeanStd(shape=(), scope='ret_rms') if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    # def step_wait(self):
    #     obs, rews, news, infos = self.venv.step_wait()
    #     self.ret = self.ret * self.gamma + rews
    #     obs = self._obfilt(obs)
    #     if self.ret_rms:
    #         self.ret_rms.update(self.ret)
    #         rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
    #     self.ret[news] = 0.
    #     return obs, rews, news, infos
    def step_wait(self):
        obs, rews0, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews0
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews0 / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        else:
            rews = rews0
        self.ret[news] = 0.
        return obs, rews0, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


def set_all_global_seeds(i):
    try:
        tf.set_random_seed(i)
        np.random.seed(i)
        random.seed(i)
    except:
        return ImportError

class LearningRateDecay(object):
    def __init__(self, v, nvalues, lr_decay_method):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues

        def constant(p):
            return 1

        def linear(p):
            return 1 - p

        lr_decay_methods = {
            'linear': linear,
            'constant': constant
        }

        self.decay = lr_decay_methods[lr_decay_method]

    def value(self):
        current_value = self.v * self.decay(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def get_value_for_steps(self, steps):
        return self.v * self.decay(steps / self.nvalues)

