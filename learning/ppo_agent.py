import numpy as np
import copy as copy
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

class PPOAgent(TFAgent):
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

    MAIN_SCOPE = "main"
    
    EXP_ACTION_FLAG = 1 << 0

    #TODO: parameter that need to tune
    TAU = 0.01
    GAMMA = 0.9
    POLICY_NOISE = 0.2
    NOISE_CLIP = 0.5
    POLICY_FREQ = 3

    def __init__(self, world, id, json_data):
        self._exp_action = False
        super().__init__(world, id, json_data)
        return

    def reset(self):
        super().reset()
        self._exp_action = False
        return

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
            self._norm_actor_tf = self._build_net_actor(actor_net_name, "actor", self._get_actor_inputs(), actor_init_output_scale, True)
            _norm_actor_target_tf = self._build_net_actor(actor_net_name, "actor_target", self._get_actor_target_inputs(), actor_init_output_scale, False)
            sample = tf.distributions.Normal(loc=0.0, scale=1.0)
            noise = tf.clip_by_value(sample.sample(1)*self.POLICY_NOISE, -self.NOISE_CLIP, self.NOISE_CLIP)
            self._norm_actor_target_noised_tf = _norm_actor_target_tf + noise
            self.current_q1 = self._build_net_critic(critic_net_name, "critic_q1", self._get_current_critic_inputs(), True)
            self._critic_q1_tf = self._build_net_critic(critic_net_name, "critic_q1", self._get_critic_inputs(), True, reuse = True)
            self.current_q2 = self._build_net_critic(critic_net_name, "critic_q2", self._get_current_critic_inputs(), True)
            self._critic_q2_tf = self._build_net_critic(critic_net_name, "critic_q2", self._get_critic_inputs(), True, reuse = True)
            self._critic_q1_target_tf = self._build_net_critic(critic_net_name, "critic_q1_target", self._get_critic_target_inputs(), False)
            self._critic_q2_target_tf = self._build_net_critic(critic_net_name, "critic_q2_target", self._get_critic_target_inputs(), False)
                
        if (self._norm_actor_tf != None):
            Logger.print("Built actor net: " + actor_net_name)

        if (self._critic_q1_tf != None):
            Logger.print("Built critic net: " + critic_net_name)
        
        self.actor_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/actor")
        self.actor_target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/actor_target")
        self.critic_q1_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q1")
        self.critic_q2_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q2")
        self.critic_q1_target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q1_target")
        self.critic_q2_target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MAIN_SCOPE+"/critic_q2_target")

        self.soft_replace = [tf.assign(t, (1-self.TAU)*t + self.TAU*e)
                                for t,e in zip(self.actor_target_param+self.critic_q1_target_param+self.critic_q2_target_param, 
                                                self.actor_param+self.critic_q1_param+self.critic_q2_param)]
        self.hard_replace = [tf.assign(t, e)
                                for t,e in zip(self.actor_target_param+self.critic_q1_target_param+self.critic_q2_target_param, 
                                                self.actor_param+self.critic_q1_param+self.critic_q2_param)]

        self.sess.run(self.hard_replace)
        
        return

    def _build_losses(self, json_data):
        q_target = self._r_ph + self.GAMMA * tf.minimum(self._critic_q1_target_tf, self._critic_q2_target_tf)

        self.td_error1 = tf.losses.mean_squared_error(labels=q_target, predictions=self.current_q1)
        self.td_error2 = tf.losses.mean_squared_error(labels=q_target, predictions=self.current_q2)

        self.a_loss = -tf.reduce_mean(self._critic_q1_tf)
         
        return

    def _build_solvers(self, json_data):
        # print("build_solvers")
        actor_stepsize = 0.001 if (self.ACTOR_STEPSIZE_KEY not in json_data) else json_data[self.ACTOR_STEPSIZE_KEY]
        actor_momentum = 0.9 if (self.ACTOR_MOMENTUM_KEY not in json_data) else json_data[self.ACTOR_MOMENTUM_KEY]
        critic_stepsize = 0.01 if (self.CRITIC_STEPSIZE_KEY not in json_data) else json_data[self.CRITIC_STEPSIZE_KEY]
        critic_momentum = 0.9 if (self.CRITIC_MOMENTUM_KEY not in json_data) else json_data[self.CRITIC_MOMENTUM_KEY]
        
        critic_q1_vars = self._tf_vars(self.MAIN_SCOPE + '/critic_q1')
        critic_q1_opt = tf.train.AdamOptimizer(learning_rate=critic_stepsize)
        self._critic_q1_grad_tf = tf.gradients(self.td_error1, critic_q1_vars)
        self._critic_q1_solver = MPISolver(self.sess, critic_q1_opt, critic_q1_vars)

        critic_q2_vars = self._tf_vars(self.MAIN_SCOPE + '/critic_q2')
        critic_q2_opt = tf.train.AdamOptimizer(learning_rate=critic_stepsize)
        self._critic_q2_grad_tf = tf.gradients(self.td_error2, critic_q2_vars)
        self._critic_q2_solver = MPISolver(self.sess, critic_q2_opt, critic_q2_vars)
        # print(self._critic_q1_grad_tf)

        actor_vars = self._tf_vars(self.MAIN_SCOPE + '/actor')
        actor_opt = tf.train.AdamOptimizer(learning_rate=actor_stepsize)
        self._actor_grad_tf = tf.gradients(self.a_loss, actor_vars)
        # print(self._actor_grad_tf, actor_vars)
        self._actor_solver = MPISolver(self.sess, actor_opt, actor_vars)
        
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
                with tf.variable_scope("output_layer", reuse=reuse):
                    a = tf.layers.dense(curr_tf, a_size, activation=tf.nn.tanh, trainable=trainable)
                norm_actor_tf = a
                # print(self._a_bound_max, self._a_bound_min)
                # bound_range = (self._a_bound_max - self._a_bound_min) / 2
                # bound_offset = (self._a_bound_max + self._a_bound_min) / 2
                # norm_actor_tf = tf.multiply(a, bound_range, "bound_scale")
                # norm_actor_tf = tf.add_n([norm_actor_tf, bound_offset], "bound_offset")
            else:
                assert False, 'Unsupported net: ' + net_name
        return norm_actor_tf
    
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

    def _get_actor_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._g_ph)
            input_tfs += [norm_g_tf]
        return input_tfs
    
    def _get_actor_target_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._n_s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._n_g_ph)
            input_tfs += [norm_g_tf]
        return input_tfs
    
    def _get_current_critic_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._g_ph)
            input_tfs += [norm_g_tf]
        input_tfs += [self._a_ph]
        return input_tfs

    def _get_critic_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._g_ph)
            input_tfs += [norm_g_tf]
        input_tfs += [self._norm_actor_tf]
        return input_tfs

    def _get_critic_target_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._n_s_ph)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self._g_norm.normalize_tf(self._n_g_ph)
            input_tfs += [norm_g_tf]
        input_tfs += [self._norm_actor_target_noised_tf]
        return input_tfs

    def _train_step(self):
        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert(start_idx == 0)
        assert(self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size) # must avoid overflow
        assert(start_idx < end_idx)

        idx = np.array(list(range(start_idx, end_idx)))        
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask) 
        
        rewards = self._fetch_batch_rewards(start_idx, end_idx)

        valid_idx = idx[end_mask]
        num_valid_idx = valid_idx.shape[0]
        
        local_sample_count = valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))

        critic_loss = 0
        actor_loss = 0

        for e in range(self.epochs):
            np.random.shuffle(valid_idx)

            for b in range(mini_batches):
                batch_idx_beg = b * self._local_mini_batch_size
                batch_idx_end = batch_idx_beg + self._local_mini_batch_size

                critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
                critic_batch = np.mod(critic_batch, num_valid_idx)

                critic_batch = valid_idx[critic_batch]

                critic_next_batch = critic_batch + 1
                mask = np.in1d(critic_next_batch, valid_idx)
                critic_batch = critic_batch[mask]
                critic_next_batch = critic_next_batch[mask]

                critic_s = self.replay_buffer.get('states', critic_batch)
                critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
                critic_a = self.replay_buffer.get('actions', critic_batch)
                critic_next_s = self.replay_buffer.get('states', critic_next_batch)
                critic_next_g = self.replay_buffer.get('goals', critic_next_batch) if self.has_goal() else None
                critic_reward = rewards[critic_batch]

                feed = {
                    self._s_ph: critic_s,
                    self._g_ph: critic_g,
                    self._a_ph: critic_a,
                    self._r_ph: critic_reward.reshape((-1,1)),
                    self._n_s_ph: critic_next_s,
                    self._n_g_ph: critic_next_g
                }
                # print(critic_s.shape, critic_a.shape, critic_reward.shape, critic_next_s.shape)
                critic_q1_loss, critic_q1_grad, critic_q2_loss, critic_q2_grad = self.sess.run([self.td_error1, self._critic_q1_grad_tf,
                                                                                                self.td_error2, self._critic_q2_grad_tf],
                                                                                                feed_dict=feed)
                self._critic_q1_solver.update(critic_q1_grad)
                self._critic_q2_solver.update(critic_q2_grad)
                curr_critic_loss = critic_q1_loss+critic_q2_loss
                critic_loss += curr_critic_loss

                if self.iter % self.POLICY_FREQ == 0:
                    # print(self.a_loss)
                    # print(self._actor_grad_tf)
                    # print(self._critic_q1_tf)
                    curr_actor_loss, actor_grad = self.sess.run([self.a_loss, self._actor_grad_tf], feed_dict=feed)
                    self._actor_solver.update(actor_grad)
                    actor_loss += np.abs(curr_actor_loss)
                    self.sess.run(self.soft_replace)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)

        critic_q1_stepsize, critic_q2_stepsize, actor_stepsize = self.sess.run([self._critic_q1_solver.optimizer._lr_t,
                                                                                self._critic_q2_solver.optimizer._lr_t,
                                                                                self._actor_solver.optimizer._lr_t])

        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Q1_Stepsize', critic_q1_stepsize)
        self.logger.log_tabular('Critic_Q2_Stepsize', critic_q2_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss) 
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)

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
            self._exp_action = True #TODO: don't know meaning of exp_action
            a, logp = self._eval_actor(s, g)
            # print(a)
            a = a[0]
            logp = logp[0]

            # print(self._a_bound_max, self._a_bound_min)
            bound_range = (self._a_bound_max - self._a_bound_min) / 2
            bound_offset = (self._a_bound_max + self._a_bound_min) / 2
            a = np.multiply(a, bound_range)
            a = np.add(a, bound_offset)
        #     print(a)
        # print("end")

        return a, logp

    def _eval_actor(self, s, g):
        s = np.reshape(s, [-1, self.get_state_size()])
        g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None
        
        feed = {
            self._s_ph : s,
            self._g_ph : g
        }
        a = self.sess.run(self._norm_actor_tf, feed_dict=feed)
        logp = np.zeros_like(a)

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