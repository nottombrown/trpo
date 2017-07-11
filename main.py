from __future__ import print_function, absolute_import, division

import logging
import tempfile
import time
from sys import argv

import prettytensor as pt
from gym import envs
from gym.spaces import Box

from utils import *

print('python main.py {}'.format(' '.join(argv)))

import argparse

parser = argparse.ArgumentParser(description='Test the new good lib.')
parser.add_argument("--task", type=str, default='InvertedDoublePendulum-v0')
parser.add_argument("--timesteps_per_batch", type=int, default=20000)
parser.add_argument("--max_pathlength", type=int, default=2000)
parser.add_argument("--n_iter", type=int, default=30)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--max_kl", type=float, default=.001)
parser.add_argument("--cg_damping", type=float, default=1e-3)

args = parser.parse_args()

algo = 'continuous_action_TRPO_nIter={}_maxKl={}_gamma={}'.format(
    args.n_iter, args.max_kl, args.gamma)

class TRPO(object):
    config = ConfigObject(
        timesteps_per_batch=args.timesteps_per_batch,
        max_pathlength=args.max_pathlength,
        gamma=args.gamma,
        n_iter=args.n_iter,
        max_kl=args.max_kl,
        cg_damping=args.cg_damping)

    def __init__(self, env):
        self.env = env
        if not isinstance(env.observation_space, Box) or\
                not isinstance(env.action_space, Box):
            print("Both the input space and the output space should be continuous.")
            print("(Probably OK to remove the requirement for the input space).")
            exit(-1)
        self.session = tf.Session()
        self.obs = obs = tf.placeholder(
            dtype, shape=[
                None, env.observation_space.shape[0]])
        act_dim = np.prod(env.action_space.shape)
        self.action = action = tf.placeholder(tf.float32, shape=[None, act_dim])
        self.advant = advant = tf.placeholder(dtype, shape=[None])
        self.old_action_dist_mu = old_action_dist_mu = tf.placeholder(dtype, shape=[None, act_dim])
        self.old_action_dist_logstd = old_action_dist_logstd = tf.placeholder(dtype, shape=[None, act_dim])

        # Create neural network.
        action_dist_mu = (pt.wrap(self.obs).
            fully_connected(64, activation_fn=tf.nn.relu).
            fully_connected(64, activation_fn=tf.nn.relu).
            fully_connected(act_dim))  # output means and logstd's

        action_dist_logstd_param = tf.Variable((.01 * np.random.randn(1, act_dim)).astype(np.float32))
        action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(action_dist_mu)[0], 1)))

        self.action_dist_mu = action_dist_mu
        self.action_dist_logstd = action_dist_logstd
        N = tf.shape(obs)[0]

        # compute probabilities of current actions and old action
        log_p_n = gauss_log_prob(action_dist_mu, action_dist_logstd, action)
        log_oldp_n = gauss_log_prob(old_action_dist_mu, old_action_dist_logstd, action)

        # proceed as before, good.
        ratio_n = tf.exp(log_p_n - log_oldp_n)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list = tf.trainable_variables()

        # Introduced the change into here: 
        kl = gauss_KL(old_action_dist_mu, old_action_dist_logstd,
            action_dist_mu, action_dist_logstd) / Nf
        ent = gauss_ent(action_dist_mu, action_dist_logstd) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)

        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = gauss_selfKL_firstfixed(action_dist_mu, action_dist_logstd) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.get_flat = GetFlat(self.session, var_list)
        self.set_from_flat = SetFromFlat(self.session, var_list)
        self.session.run(tf.initialize_variables(var_list))
        self.vf = LinearVF()

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd =\
            self.session.run([self.action_dist_mu, self.action_dist_logstd], {self.obs: obs})

        act = action_dist_mu + np.exp(action_dist_logstd) * np.random.randn(*action_dist_logstd.shape)

        return act.ravel(),\
            ConfigObject(action_dist_mu=action_dist_mu,
                action_dist_logstd=action_dist_logstd)

    def learn(self):
        config = self.config
        start_time = time.time()
        timesteps_elapsed = 0

        for i in range(1, config.n_iter):
            # Generating paths.
            paths = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch,
                render=False)  # (i % render_freq) == 0)

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config.gamma)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist_mu = np.concatenate([path["action_dists_mu"] for path in paths])
            action_dist_logstd = np.concatenate([path["action_dists_logstd"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()
            advant_n /= (advant_n.std() + 1e-8)

            # Computing baseline function for next iter.
            self.vf.fit(paths)

            feed = {self.obs: obs_n,
                self.action: action_n,
                self.advant: advant_n,
                self.old_action_dist_mu: action_dist_mu,
                self.old_action_dist_logstd: action_dist_logstd}

            theta_prev = self.get_flat()

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.session.run(self.fvp, feed) + p * config.cg_damping

            g = self.session.run(self.pg, feed_dict=feed)
            stepdir = conjugate_gradient(fisher_vector_product, -g)
            shs = (.5 * stepdir.dot(fisher_vector_product(stepdir)))
            assert shs > 0

            lm = np.sqrt(shs / config.max_kl)

            fullstep = stepdir / lm

            theta = theta_prev + fullstep
            self.set_from_flat(theta)

            surrogate_loss, kl_old_new, entropy = self.session.run(self.losses, feed_dict=feed)

            ep_rewards = np.array(
                [path["rewards"].sum() for path in paths])

            stats = {}
            timesteps_elapsed += sum([len(path["rewards"]) for path in paths])
            stats["Total number of times steps"] = timesteps_elapsed
            stats["Average sum of rewards per episode"] = ep_rewards.mean()
            stats["Entropy"] = entropy
            stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
            stats["KL between old and new distribution"] = kl_old_new
            stats["Surrogate loss"] = surrogate_loss
            print("\n********** Iteration {} ************".format(i))
            for k, v in stats.items():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if entropy != entropy:
                exit(-1)

logging.getLogger().setLevel(logging.DEBUG)

env = envs.make(args.task)

agent = TRPO(env)
agent.learn()

from sys import argv

print('python {}'.format(' '.join(argv)))
