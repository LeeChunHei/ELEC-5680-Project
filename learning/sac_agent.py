import time
import numpy as np
import copy as copy
from pkg_resources import resource_stream
import tensorflow as tf

# from learning.pg_agent import PGAgent
from learning.tf_agent import TFAgent
from learning.solvers.mpi_solver import MPISolver
import learning.tf_util as TFUtil
import learning.rl_util as RLUtil
import learning.nets.fc_2layers_1024units as fc_2layers_1024units
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.math_util as MathUtil
from env.action_space import ActionSpace
from env.env import Env

'''
Proximal Policy Optimization Agent
'''

class SACAgent(TFAgent):
    NAME = "PPO"
    EPOCHS_KEY = "Epochs"
    BATCH_SIZE_KEY = "BatchSize"
    RATIO_CLIP_KEY = "RatioClip"
    NORM_ADV_CLIP_KEY = "NormAdvClip"
    TD_LAMBDA_KEY = "TDLambda"
    TAR_CLIP_FRAC = "TarClipFrac"
    
    ADV_EPS = 1e-5

    ACTOR_NET_KEY = 'ActorNet'
    ACTOR_STEPSIZE_KEY = 'ActorStepsize'
    ACTOR_MOMENTUM_KEY = 'ActorMomentum'
    ACTOR_WEIGHT_DECAY_KEY = 'ActorWeightDecay'
    ACTOR_INIT_OUTPUT_SCALE_KEY = 'ActorInitOutputScale'

    CRITIC_NET_KEY = 'CriticNet'
    CRITIC_STEPSIZE_KEY = 'CriticStepsize'
    CRITIC_MOMENTUM_KEY = 'CriticMomentum'
    CRITIC_WEIGHT_DECAY_KEY = 'CriticWeightDecay'

    ALPHA_STEPSIZE_KEY = 'AlphaStepsize'

    MAIN_SCOPE = "main"
    
    EXP_ACTION_FLAG = 1 << 0

    #TODO: parameter that need to tune
    TAU = 0.005
    GAMMA = 0.9
    POLICY_NOISE = 0.2
    NOISE_CLIP = 0.5
    POLICY_FREQ = 2
    LOG_SIG_MIN = -20
    LOG_SIG_MAX = 2
    EPSILON = 1e-6
    EPS = 1e-8
    ALPHA = 0.2
    TARGET_UPDATE_INTERVAL = 1

    def __init__(self, world, id, json_data):
        self._exp_action = False
        super().__init__(world, id, json_data)
        return

    def reset(self):
        super().reset()
        self._exp_action = False
        return

    def _store_path(self, path):
        path_id = super()._store_path(path)

        valid_path = (path_id != MathUtil.INVALID_IDX)
        if (valid_path):
            for i in range(len(path.states)-2):
                if path.flags[i] == self.EXP_ACTION_FLAG:
                    state = path.states[i]
                    next_state = path.states[i+1]
                    action = path.actions[i]
                    amp_obs = path.amp_obs_agent[i]
                    self.sac_replay_buffer.add(state, action, next_state, amp_obs)
        return path_id

    def _load_params(self, json_data):
        super()._load_params(json_data)

        self.epochs = 1 if (self.EPOCHS_KEY not in json_data) else json_data[self.EPOCHS_KEY]
        self.batch_size = 1024 if (self.BATCH_SIZE_KEY not in json_data) else json_data[self.BATCH_SIZE_KEY]
        self.ratio_clip = 0.2 if (self.RATIO_CLIP_KEY not in json_data) else json_data[self.RATIO_CLIP_KEY]
        self.norm_adv_clip = 5 if (self.NORM_ADV_CLIP_KEY not in json_data) else json_data[self.NORM_ADV_CLIP_KEY]
        self.td_lambda = 0.95 if (self.TD_LAMBDA_KEY not in json_data) else json_data[self.TD_LAMBDA_KEY]
        self.tar_clip_frac = -1 if (self.TAR_CLIP_FRAC not in json_data) else json_data[self.TAR_CLIP_FRAC]

        num_procs = MPIUtil.get_num_procs()
        self._local_batch_size = int(np.ceil(self.batch_size / num_procs))
        min_replay_size = 2 * self._local_batch_size # needed to prevent buffer overflow
        assert(self.replay_buffer_size > min_replay_size)

        self.replay_buffer_size = np.maximum(min_replay_size, self.replay_buffer_size)

        td3_buffer_size = 10000
        td3_buffer_size = np.maximum(min_replay_size, td3_buffer_size)

        self.sac_replay_buffer = SACReplayBuffer(self.get_state_size(), self.get_action_size(), self._get_amp_obs_size(), td3_buffer_size) #self.replay_buffer_size

        return

    def _build_nets(self, json_data):
        assert self.ACTOR_NET_KEY in json_data
        assert self.CRITIC_NET_KEY in json_data

        actor_net_name = json_data[self.ACTOR_NET_KEY]
        critic_net_name = json_data[self.CRITIC_NET_KEY]
        actor_init_output_scale = 1 if (self.ACTOR_INIT_OUTPUT_SCALE_KEY not in json_data) else json_data[self.ACTOR_INIT_OUTPUT_SCALE_KEY]

        s_size = self.get_state_size()
        g_size = self.get_goal_size()
        a_size = self.get_action_size()

        # setup input tensors
        self._s_ph = tf.placeholder(tf.float32, shape=[None, s_size], name="s")
        self._g_ph = tf.placeholder(tf.float32, shape=([None, g_size] if self.has_goal() else None), name="g")
        self._a_ph = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
        self._r_ph = tf.placeholder(tf.float32, shape=[None, 1], name="r")
        self._n_s_ph = tf.placeholder(tf.float32, shape=[None, s_size], name="n_s")
        self._n_g_ph = tf.placeholder(tf.float32, shape=([None, g_size] if self.has_goal() else None), name="n_g")
        # self._old_logp_ph = tf.placeholder(tf.float32, shape=[None], name="old_logp")
        # self._tar_val_ph = tf.placeholder(tf.float32, shape=[None], name="tar_val")
        # self._adv_ph = tf.placeholder(tf.float32, shape=[None], name="adv")

        with tf.variable_scope(self.MAIN_SCOPE):
            self._actor_action_tf, self._actor_log_prob_tf, self._actor_mean_tf = self._build_net_actor(actor_net_name, "actor", self._get_actor_inputs(), actor_init_output_scale, True)
            self._actor_next_action_tf, self._actor_next_log_prob_tf, self._actor_next_mean_tf = self._build_net_actor(actor_net_name, "actor", self._get_next_actor_inputs(), actor_init_output_scale, True, reuse=True)
            
            self._critic_q1_tf = self._build_net_critic(critic_net_name, "critic_q1", self._get_critic_inputs(), True)
            self._critic_q2_tf = self._build_net_critic(critic_net_name, "critic_q2", self._get_critic_inputs(), True)
            self._critic_q1_pi_tf = self._build_net_critic(critic_net_name, "critic_q1", self._get_critic_pi_inputs(), True, reuse=True)
            self._critic_q2_pi_tf = self._build_net_critic(critic_net_name, "critic_q2", self._get_critic_pi_inputs(), True, reuse=True)
            self._critic_q1_target_tf = self._build_net_critic(critic_net_name, "critic_q1_target", self._get_critic_target_inputs(), False)
            self._critic_q2_target_tf = self._build_net_critic(critic_net_name, "critic_q2_target", self._get_critic_target_inputs(), False)
            
            # self._target_entropy = -tf.reduce_prod(tf.constant((a_size,), dtype=tf.float32), name="target_entropy").eval()
            # print("target_entropy:",self._target_entropy)
            # with tf.variable_scope("log_alpha", reuse=False):
            #     self._log_alpha = tf.get_variable("log_alpha", initializer=np.zeros(1).astype(np.float32))
            # print(tf.trainable_variables())
            # print("next", self._log_alpha)
            self.alpha = tf.constant(0.2, name="alpha")# tf.exp(self._log_alpha, name="alpha")

        if (self._actor_action_tf != None):
            Logger.print("Built actor net: " + actor_net_name)

        if (self._critic_q1_tf != None):
            Logger.print("Built critic net: " + critic_net_name)

        self.actor_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/actor")
        self.critic_q1_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q1")
        self.critic_q2_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q2")
        self.critic_q1_target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q1_target")
        self.critic_q2_target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q2_target")

        self.soft_replace = [tf.assign(t, (1-self.TAU)*t + self.TAU*e)
                                for t,e in zip(self.critic_q1_target_param+self.critic_q2_target_param, 
                                               self.critic_q1_param+self.critic_q2_param)]
        self.hard_replace = [tf.assign(t, e)
                                for t,e in zip(self.critic_q1_target_param+self.critic_q2_target_param, 
                                               self.critic_q1_param+self.critic_q2_param)]

        self.sess.run(self.hard_replace)
        
        return

    def _build_losses(self, json_data):

        min_qf_next_target = tf.minimum(self._critic_q1_target_tf, self._critic_q2_target_tf) - self.alpha * self._actor_next_log_prob_tf
        next_q_value = tf.stop_gradient(self._r_ph + self.GAMMA * (min_qf_next_target))
        
        self._qf1_loss = tf.losses.mean_squared_error(labels=next_q_value, predictions=self._critic_q1_tf)
        self._qf2_loss = tf.losses.mean_squared_error(labels=next_q_value, predictions=self._critic_q2_tf)

        min_qf_pi = tf.minimum(self._critic_q1_pi_tf, self._critic_q2_pi_tf)
        self._policy_loss = tf.reduce_mean((self.alpha*self._actor_log_prob_tf) - min_qf_pi)

        # _alpha_loss = -1.0 * (self.alpha * tf.stop_gradient(self._actor_log_prob_tf + self._target_entropy))
        # self._alpha_loss = tf.reduce_mean(_alpha_loss)

        return

    def _build_action_bound_loss(self, _actor_tf):
        norm_a_bound_min = self._a_norm.normalize_tf(self._a_bound_min)
        norm_a_bound_max = self._a_norm.normalize_tf(self._a_bound_max)

        min_violation = tf.minimum(tf.subtract(_actor_tf, norm_a_bound_min), 0)
        max_violation = tf.maximum(tf.subtract(_actor_tf, norm_a_bound_max), 0)
        loss = 0.5*(tf.reduce_mean(tf.reduce_sum(tf.square(min_violation), axis=-1)) + 
                    tf.reduce_mean(tf.reduce_sum(tf.square(max_violation), axis=-1)))
        
        return loss

    def _build_solvers(self, json_data):
        # print("build_solvers")
        actor_stepsize = 0.001 if (self.ACTOR_STEPSIZE_KEY not in json_data) else json_data[self.ACTOR_STEPSIZE_KEY]
        actor_momentum = 0.9 if (self.ACTOR_MOMENTUM_KEY not in json_data) else json_data[self.ACTOR_MOMENTUM_KEY]
        critic_stepsize = 0.01 if (self.CRITIC_STEPSIZE_KEY not in json_data) else json_data[self.CRITIC_STEPSIZE_KEY]
        critic_momentum = 0.9 if (self.CRITIC_MOMENTUM_KEY not in json_data) else json_data[self.CRITIC_MOMENTUM_KEY]
        alpha_stepsize = actor_stepsize if (self.ALPHA_STEPSIZE_KEY not in json_data) else json_data[self.ALPHA_STEPSIZE_KEY]
        
        critic_q1_vars = self._tf_vars(self.MAIN_SCOPE + '/critic_q1')
        critic_q1_opt = tf.train.AdamOptimizer(learning_rate=critic_stepsize)
        self._critic_q1_grad_tf = tf.gradients(self._qf1_loss, critic_q1_vars)
        self._critic_q1_solver = MPISolver(self.sess, critic_q1_opt, critic_q1_vars)

        critic_q2_vars = self._tf_vars(self.MAIN_SCOPE + '/critic_q2')
        critic_q2_opt = tf.train.AdamOptimizer(learning_rate=critic_stepsize)
        self._critic_q2_grad_tf = tf.gradients(self._qf2_loss, critic_q2_vars)
        self._critic_q2_solver = MPISolver(self.sess, critic_q2_opt, critic_q2_vars)
        # print(self._critic_q1_grad_tf)

        actor_vars = self._tf_vars(self.MAIN_SCOPE + '/actor')
        actor_opt = tf.train.AdamOptimizer(learning_rate=actor_stepsize)
        self._actor_grad_tf = tf.gradients(self._policy_loss, actor_vars)
        # print(self._actor_grad_tf, actor_vars)
        self._actor_solver = MPISolver(self.sess, actor_opt, actor_vars)
        
        # log_alpha_vars = self._tf_vars(self.MAIN_SCOPE + '/log_alpha')
        # alpha_opt = tf.train.AdamOptimizer(learning_rate=alpha_stepsize)
        # self._alpha_grad_tf = tf.gradients(self._alpha_loss, log_alpha_vars)
        # self._alpha_solver = MPISolver(self.sess, alpha_opt, log_alpha_vars)

        return

    def _build_net_actor(self, net_name, scope, input_tfs, init_output_scale, trainable, reuse=False):
        a_size = self.get_action_size()
        with tf.variable_scope(scope, reuse=reuse):
            if (net_name == fc_2layers_1024units.NAME):
                layers = [1024, 512]
                activation = tf.nn.relu

                input_tf = tf.concat(axis=-1, values=input_tfs)
                curr_tf = input_tf
                for i, size in enumerate(layers):
                    with tf.variable_scope(str(i), reuse=reuse):
                        curr_tf = tf.layers.dense(inputs=curr_tf,
                                                  units=size,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  activation = activation,
                                                  trainable=trainable)
                with tf.variable_scope("mean_layer", reuse=reuse):
                    _mean_tf = tf.layers.dense(curr_tf, a_size, trainable=trainable, 
                                                kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale),
                                                bias_initializer=tf.zeros_initializer())
                with tf.variable_scope("logstd_layer", reuse=reuse):
                    _logstd_tf = tf.layers.dense(curr_tf, a_size, trainable=trainable, 
                                                kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale),
                                                bias_initializer=tf.zeros_initializer())    #deepmimic use different bias_init
                    _logstd_tf = tf.clip_by_value(_logstd_tf, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
            else:
                assert False, 'Unsupported net: ' + net_name

        # std = tf.exp(_logstd_tf)
        # rsample_normal = tf.distributions.Normal(_mean_tf, std)
        # x_t = rsample_normal.sample()
        # y_t = tf.tanh(x_t)
        bound_range = (self._a_bound_max - self._a_bound_min) / 2
        bound_offset = (self._a_bound_max + self._a_bound_min) / 2
        # action_tf = y_t * bound_range + bound_offset
        # log_prob = rsample_normal.log_prob(x_t) - tf.log(bound_range * (1 - tf.pow(y_t, 2)) + self.EPSILON)
        # log_prob_tf = tf.reduce_sum(log_prob, 1, keepdims=True)
        # mean_tf = tf.tanh(_mean_tf) * bound_range + bound_offset

        def gaussian_likelihood(x, mu, log_std):
            pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+self.EPS))**2 + 2*log_std + np.log(2*np.pi))
            return tf.reduce_sum(pre_sum, axis=1)

        #spining up policy
        mu = _mean_tf
        std = tf.exp(_logstd_tf)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, _logstd_tf)
        #apply squashing
        logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        #scale
        mu = mu * bound_range + bound_offset
        pi = pi * bound_range + bound_offset

        return pi, logp_pi, mu #action_tf, log_prob_tf, mean_tf
    
    def _build_net_critic(self, net_name, scope, input_tfs, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            if (net_name == fc_2layers_1024units.NAME):
                layers = [1024, 512]
                activation = tf.nn.relu

                input_tf = tf.concat(axis=-1, values=input_tfs)
                curr_tf = input_tf
                for i, size in enumerate(layers):
                    with tf.variable_scope(str(i), reuse=reuse):
                        curr_tf = tf.layers.dense(inputs=curr_tf,
                                                  units=size,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  activation = activation,
                                                  trainable=trainable)
                with tf.variable_scope("output_layer", reuse=reuse):
                    c = tf.layers.dense(curr_tf, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
            else:
                assert False, 'Unsupported net: ' + net_name
        return c

    def _build_net_value(self, net_name, scope, input_tfs, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            if (net_name == fc_2layers_1024units.NAME):
                layers = [1024, 512]
                activation = tf.nn.relu

                input_tf = tf.concat(axis=-1, values=input_tfs)
                curr_tf = input_tf
                for i, size in enumerate(layers):
                    with tf.variable_scope(str(i), reuse=reuse):
                        curr_tf = tf.layers.dense(inputs=curr_tf,
                                                  units=size,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  activation = activation,
                                                  trainable=trainable)
                with tf.variable_scope("output_layer", reuse=reuse):
                    c = tf.layers.dense(curr_tf, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
            else:
                assert False, 'Unsupported net: ' + net_name
        return c

    def _get_scaled_action(self, action_tf):
        pass

    def _get_unscaled_action(self, action_tf):
        pass

    def _get_actor_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._g_ph)
            input_tfs += [norm_g_tf]
        return input_tfs
    
    def _get_next_actor_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._n_s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._n_g_ph)
            input_tfs += [norm_g_tf]
        return input_tfs
    
    def _get_critic_pi_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._g_ph)
            input_tfs += [norm_g_tf]
        input_tfs += [self._actor_action_tf]
        return input_tfs

    def _get_critic_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._g_ph)
            input_tfs += [norm_g_tf]
        input_tfs += [self._a_ph]
        return input_tfs

    def _get_critic_target_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._n_s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._n_g_ph)
            input_tfs += [norm_g_tf]
        input_tfs += [self._actor_next_action_tf]
        return input_tfs

    def _calc_reward(self, obs_agent):
        disc_r, _ = self._calc_disc_reward(obs_agent)
        
        disc_r *= self._reward_scale
        self._disc_reward_mean = np.mean(disc_r)
        self._disc_reward_std = np.std(disc_r)
        
        if (self._enable_amp_task_reward()):
            assert False, "should not enable amp task reward"
        else:
            r = disc_r
        
        curr_reward_min = np.amin(r)
        curr_reward_max = np.amax(r)
        self._reward_min = np.minimum(self._reward_min, curr_reward_min)
        self._reward_max = np.maximum(self._reward_max, curr_reward_max)
        reward_data = np.array([self._reward_min, -self._reward_max])
        reward_data = MPIUtil.reduce_min(reward_data)

        self._reward_min = reward_data[0]
        self._reward_max = -reward_data[1]

        return r

    def _train_step(self):
        # start_idx = self.replay_buffer.buffer_tail
        # end_idx = self.replay_buffer.buffer_head
        # assert(start_idx == 0)
        # assert(self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size) # must avoid overflow
        # assert(start_idx < end_idx)

        # idx = np.array(list(range(start_idx, end_idx)))        
        # end_mask = self.replay_buffer.is_path_end(idx)
        # end_mask = np.logical_not(end_mask) 
        
        amp_obs = self.sac_replay_buffer.amp_obs[:self.sac_replay_buffer.size]

        rewards = self._calc_reward(amp_obs)

        # valid_idx = idx[end_mask]
        # num_valid_idx = valid_idx.shape[0]
        
        local_sample_count = self.sac_replay_buffer.size #valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))

        valid_idx = np.array(list(range(0,local_sample_count)))
        num_valid_idx = len(valid_idx)

        critic_loss = 0
        actor_loss = 0
        alpha_loss = 0

        # print("shape")
        # print(start_idx, end_idx)
        # print(self.replay_buffer.buffers['states'].shape)
        # print(rewards.shape)

        for e in range(self.epochs):
            np.random.shuffle(valid_idx)

            for b in range(mini_batches):
                batch_idx_beg = b * self._local_mini_batch_size
                batch_idx_end = batch_idx_beg + self._local_mini_batch_size

                critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
                critic_batch = np.mod(critic_batch, num_valid_idx)

                critic_batch = valid_idx[critic_batch]

                # critic_next_batch = critic_batch + 1
                # mask = np.in1d(critic_next_batch, valid_idx)
                # critic_batch = critic_batch[mask]
                # critic_next_batch = critic_next_batch[mask]

                # critic_s = self.replay_buffer.get('states', critic_batch)
                # critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
                # critic_a = self.replay_buffer.get('actions', critic_batch)
                # critic_next_s = self.replay_buffer.get('states', critic_next_batch)
                # critic_next_g = self.replay_buffer.get('goals', critic_next_batch) if self.has_goal() else None
                # critic_reward = rewards[critic_batch]

                critic_s, critic_a, critic_next_s, _ = self.sac_replay_buffer.sample(critic_batch)
                critic_reward = rewards[critic_batch]

                feed = {
                    self._s_ph: critic_s,
                    self._g_ph: None,#critic_g,
                    self._a_ph: critic_a,
                    self._r_ph: critic_reward.reshape((-1,1)),
                    self._n_s_ph: critic_next_s,
                    self._n_g_ph: None,#critic_next_g
                }
                # print(critic_s.shape, critic_a.shape, critic_reward.shape, critic_next_s.shape)
                qf1_grad, qf1_loss, qf2_grad, qf2_loss = self.sess.run([self._critic_q1_grad_tf, self._qf1_loss,
                                                                        self._critic_q2_grad_tf, self._qf2_loss],
                                                                        feed_dict=feed)

                self._critic_q1_solver.update(qf1_grad)
                self._critic_q2_solver.update(qf2_grad)
                curr_critic_loss = qf1_loss+qf2_loss
                critic_loss += curr_critic_loss

                policy_grad, policy_loss = self.sess.run([self._actor_grad_tf, self._policy_loss], feed_dict=feed)
                self._actor_solver.update(policy_grad)
                actor_loss += abs(policy_loss)

                # alpha_grad, curr_alpha_loss = self.sess.run([self._alpha_grad_tf, self._alpha_loss], feed_dict=feed)
                # if np.isnan(alpha_grad) or np.isnan(curr_alpha_loss):
                #     print("grad or loss nan")
                #     print(alpha_grad, curr_alpha_loss)
                #     while True:
                #         pass
                # self._alpha_solver.update(alpha_grad)
                # if np.isnan(self.alpha.eval()):
                #     print("alpha nan")
                #     print(alpha_grad, curr_alpha_loss)
                #     while True:
                #         pass
                # alpha_loss += curr_alpha_loss

        if self.iter % self.TARGET_UPDATE_INTERVAL == 0:
            self.sess.run(self.soft_replace)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches
        alpha_loss /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)
        alpha_loss = MPIUtil.reduce_avg(alpha_loss)

        critic_q1_stepsize, critic_q2_stepsize, actor_stepsize = self.sess.run([self._critic_q1_solver.optimizer._lr_t,
                                                                                self._critic_q2_solver.optimizer._lr_t,
                                                                                self._actor_solver.optimizer._lr_t])

        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Q1_Stepsize', critic_q1_stepsize)
        self.logger.log_tabular('Critic_Q2_Stepsize', critic_q2_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss) 
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
        self.logger.log_tabular('Alpha_Loss', alpha_loss)
        self.logger.log_tabular('Alpha', self.alpha.eval())

    def _train(self):
        super()._train()
        self.replay_buffer.clear()
        return
    
    def _fetch_batch_rewards(self, start_idx, end_idx):
        print("should not call this reward")
        rewards = self.replay_buffer.get_all("rewards")[start_idx:end_idx]
        return rewards

    def _get_iters_per_update(self):
        return 1

    def _valid_train_step(self):
        samples = self.replay_buffer.get_current_size()
        exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
        return (samples >= self._local_batch_size) and (exp_samples >= self._local_mini_batch_size)

    def _check_action_space(self):
        action_space = self.get_action_space()
        return action_space == ActionSpace.Continuous

    # def _update_critic(self, s, g, tar_vals):
    #     feed = {
    #         self._s_ph: s,
    #         self._g_ph: g,
    #         self._tar_val_ph: tar_vals
    #     }

    #     loss, grads = self.sess.run([self._critic_loss_tf, self._critic_grad_tf], feed)
    #     self._critic_solver.update(grads)
    #     return loss
    
    # def _update_actor(self, s, g, a, logp, adv):
    #     feed = {
    #         self._s_ph: s,
    #         self._g_ph: g,
    #         self._a_ph: a,
    #         self._adv_ph: adv,
    #         self._old_logp_ph: logp
    #     }

    #     loss, grads, clip_frac = self.sess.run([self._actor_loss_tf, self._actor_grad_tf,
    #                                             self._clip_frac_tf], feed)
    #     self._actor_solver.update(grads)

    #     return loss, clip_frac

    def _decide_action(self, s, g):
        # print("decide_action")
        with self.sess.as_default(), self.graph.as_default():
            self._exp_action = self._enable_stoch_policy()# and MathUtil.flip_coin(self.exp_params_curr.rate)
            a, logp = self._eval_actor(s, g)
            # print(a)
            a = a[0]
            logp = logp[0]

            a = np.clip(a, self._a_bound_min, self._a_bound_max)
            # # print(self._a_bound_max, self._a_bound_min)
            # bound_range = (self._a_bound_max - self._a_bound_min) / 2
            # bound_offset = (self._a_bound_max + self._a_bound_min) / 2
            # a = np.multiply(a, bound_range)
            # a = np.add(a, bound_offset)
        #     print(a)
        # print("end")

        return a, logp

    def _enable_stoch_policy(self):
        return self.enable_training and (self._mode == self.Mode.TRAIN or self._mode == self.Mode.TRAIN_END)

    def _eval_actor(self, s, g):
        s = np.reshape(s, [-1, self.get_state_size()])
        g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None
        
        feed = {
            self._s_ph : s,
            self._g_ph : g
        }
        if self._exp_action:
            a, logp = self.sess.run([self._actor_action_tf, self._actor_log_prob_tf], feed_dict=feed)
        else:
            a, logp = self.sess.run([self._actor_mean_tf, self._actor_log_prob_tf], feed_dict=feed)

        return a, logp

    def _build_replay_buffer(self, buffer_size):
        super()._build_replay_buffer(buffer_size)
        self.replay_buffer.add_filter_key(self.EXP_ACTION_FLAG)
        return

    def _record_flags(self):
        flags = int(0)
        if (self._exp_action):
            flags = flags | self.EXP_ACTION_FLAG
        return flags

    def _initialize_vars(self):
        super()._initialize_vars()
        self._sync_solvers()
        return

    def _sync_solvers(self):
        self._actor_solver.sync()
        self._critic_q1_solver.sync()
        self._critic_q2_solver.sync()
        return

class SACReplayBuffer():
    def __init__(self, state_dim, action_dim, amp_obs_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.amp_obs = np.zeros((max_size, amp_obs_dim))

    def add(self, state, action, next_state, amp_obs):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.amp_obs[self.ptr] = amp_obs

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch):#_size):
        ind = batch
        # ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.amp_obs[ind],
        )
