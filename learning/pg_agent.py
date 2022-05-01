from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from learning.tf_agent import TFAgent
# from learning.solvers.mpi_solver import MPISolver
# import learning.nets.net_builder as NetBuilder
# from learning.tf_distribution_gaussian_diag import TFDistributionGaussianDiag
# import learning.rl_util as RLUtil
from env import Env

'''
Policy Gradient Agent
'''

class PGAgent(TFAgent):
    NAME = 'PG'

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

    def __init__(self, json_data, env, writer): 
        self._exp_action = False
        super().__init__(json_data, env, writer)
        return

    def reset(self):
        super().reset()
        self._exp_action = False
        return

    def _check_action_space(self):
        action_space = self.get_action_space()
        return action_space == 1#Continuous

    def _load_params(self, json_data):
        super()._load_params(json_data)
        self.val_min, self.val_max = self._calc_val_bounds(self.discount)
        self.val_fail, self.val_succ = self._calc_term_vals(self.discount)
        return

    def save_model(self, out_path):
        super().save_model(out_path)
        checkpoint = {}
        checkpoint['actor'] = self._norm_a_pd_tf.state_dict()
        checkpoint['critic'] = self._critic_tf.state_dict()
        checkpoint['actor_opt'] = self._actor_opt.state_dict()
        checkpoint['critic_opt'] = self._critic_opt.state_dict()
        torch.save(checkpoint, out_path+"_pg_agent")
        #TODO: need to see what to save, and need to be called by upper class

    def load_model(self, in_path):
        super().save_model(in_path)
        checkpoint = torch.load(in_path+"_pg_agent")
        self._norm_a_pd_tf.load_state_dict(checkpoint["actor"])
        self._critic_tf.load_state_dict(checkpoint["critic"])
        self._actor_opt.load_state_dict(checkpoint['actor_opt'])
        self._critic_opt.load_state_dict(checkpoint['critic_opt'])

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
        # self._s_ph = tf.placeholder(tf.float32, shape=[None, s_size], name="s") # observations
        # self._tar_val_ph = tf.placeholder(tf.float32, shape=[None], name="tar_val") # target value s
        # self._adv_ph = tf.placeholder(tf.float32, shape=[None], name="adv") # advantage
        # self._a_ph = tf.placeholder(tf.float32, shape=[None, a_size], name="a") # target actions
        # self._g_ph = tf.placeholder(tf.float32, shape=([None, g_size] if self.has_goal() else None), name="g") # goals

        self._norm_a_pd_tf = self._build_net_actor(actor_net_name, s_size+g_size, actor_init_output_scale)
        self._critic_tf = self._build_net_critic(critic_net_name, s_size+g_size)

        if (self.actor_tf != None):
            print('Built actor net: ' + actor_net_name)

        if (self.critic_tf != None):
            print('Built critic net: ' + critic_net_name)
            
        return

    def _build_losses(self, json_data):
        self.actor_bound_loss_weight = 10.0
        
        return

    def _build_solvers(self, json_data):
        actor_stepsize = 0.001 if (self.ACTOR_STEPSIZE_KEY not in json_data) else json_data[self.ACTOR_STEPSIZE_KEY]
        actor_momentum = 0.9 if (self.ACTOR_MOMENTUM_KEY not in json_data) else json_data[self.ACTOR_MOMENTUM_KEY]
        critic_stepsize = 0.01 if (self.CRITIC_STEPSIZE_KEY not in json_data) else json_data[self.CRITIC_STEPSIZE_KEY]
        critic_momentum = 0.9 if (self.CRITIC_MOMENTUM_KEY not in json_data) else json_data[self.CRITIC_MOMENTUM_KEY]
        actor_weight_decay = 0 if (self.ACTOR_WEIGHT_DECAY_KEY not in json_data) else json_data[self.ACTOR_WEIGHT_DECAY_KEY]
        critic_weight_decay = 0 if (self.CRITIC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.CRITIC_WEIGHT_DECAY_KEY]
        
        # critic_vars = self._tf_vars(self.MAIN_SCOPE + '/critic')
        self._critic_opt = torch.optim.SGD(self._critic_tf.parameters(), lr=critic_stepsize, momentum=critic_momentum, weight_decay=critic_weight_decay)
        # self._critic_grad_tf = tf.gradients(self._critic_loss_tf, critic_vars)
        # self._critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

        # actor_vars = self._tf_vars(self.MAIN_SCOPE + '/actor')
        self._actor_opt = torch.optim.SGD(self._norm_a_pd_tf.parameters(), lr=actor_stepsize, momentum=actor_momentum, weight_decay=actor_weight_decay)
        # self._actor_grad_tf = tf.gradients(self._actor_loss_tf, actor_vars)
        # self._actor_solver = MPISolver(self.sess, actor_opt, actor_vars)

        return

    def _build_net_actor(self, net_name, input_dim, init_output_scale, reuse=False):
        actor_input_size = input_dim
        action_size = self.get_action_size()
        if net_name == "fc_2layers_1024units":
            norm_a_pd_tf = Actor(actor_input_size, action_size, [1024,512], init_output_scale, self.exp_params_curr.noise)
        else:
            assert False, net_name+" not supported"

        return norm_a_pd_tf
    
    def _build_net_critic(self, net_name, input_dim, reuse=False):
        out_size = 1
        critic_input_size = input_dim
        if net_name == "fc_2layers_1024units":
            val_tf = Critic(critic_input_size, out_size, [1024,512])
        else:
            assert False, net_name+" not supported"

        return val_tf
    
    def _get_actor_inputs(self, s, g):
        norm_actor_input = []
        norm_s_tf = self._s_norm.normalize_tf(s)
        norm_actor_input.append(norm_s_tf)
        if self.has_goal():
            norm_g_tf = self._g_norm.normalize_tf(g)
            norm_actor_input.append(norm_g_tf)
        norm_actor_input = torch.concat(norm_actor_input, 1)
        return norm_actor_input
    
    def _get_critic_inputs(self, s, g):
        norm_critic_input = []
        norm_s_tf = self._s_norm.normalize_tf(s)
        # print("s_norm_mean", self._s_norm.mean[:10], self._s_norm.std[:10], self._s_norm.mean_tf[:10], self._s_norm.std_tf[:10])
        norm_critic_input.append(norm_s_tf)
        if self.has_goal():
            norm_g_tf = self._g_norm.normalize_tf(g)
            norm_critic_input.append(norm_g_tf)
        norm_critic_input = torch.concat(norm_critic_input, 1)
        return norm_critic_input

    def _build_action_bound_loss(self, norm_a_pd_tf, mean):
        norm_a_bound_min = self._a_norm.normalize(self._a_bound_min)
        norm_a_bound_max = self._a_norm.normalize(self._a_bound_max)
        
        if (isinstance(norm_a_pd_tf, Actor)):
            logstd_min = -np.inf
            logstd_max = np.inf
            norm_a_logstd_min = logstd_min * np.ones_like(norm_a_bound_min)
            norm_a_logstd_max = logstd_max * np.ones_like(norm_a_bound_max)
            norm_a_bound_min = np.concatenate([norm_a_bound_min, norm_a_logstd_min], axis=-1)
            norm_a_bound_max = np.concatenate([norm_a_bound_max, norm_a_logstd_max], axis=-1)
        
        a_bound_loss = norm_a_pd_tf.param_bound_loss(norm_a_bound_min, norm_a_bound_max, mean)
        return a_bound_loss
    
    def _initialize_vars(self):
        super()._initialize_vars()
        self._sync_solvers()
        return

    def _sync_solvers(self):
        # self._actor_solver.sync()
        # self._critic_solver.sync()
        return

    def _decide_action(self, s, g):
        flip_coin = np.random.binomial(1, self.exp_params_curr.rate, 1)
        flip_coin = flip_coin[0] == 1
        self._exp_action = self._enable_stoch_policy() and flip_coin

        a, logp = self._eval_actor(s, g, self._exp_action)
        a = a[0]
        logp = logp[0]

        return a, logp

    def _enable_stoch_policy(self):
        return self.enable_training and (self._mode == self.Mode.TRAIN or self._mode == self.Mode.TRAIN_END)

    def _eval_actor(self, s, g, exp_action):
        s = np.reshape(s, [-1, self.get_state_size()])
        g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None
        
        #prepare norm actor input
        norm_actor_input = self._get_actor_inputs(s, g)
        if exp_action:
            mean = self._norm_a_pd_tf.mean(norm_actor_input)
            sample_norm_a_tf = self._norm_a_pd_tf.sample(mean)
            a = self._a_norm.unnormalize_tf(sample_norm_a_tf)
            logp = self._norm_a_pd_tf.logp(sample_norm_a_tf, mean)
        else:
            mean = self._norm_a_pd_tf.mean(norm_actor_input)
            a = self._a_norm.unnormalize_tf(mean)
            logp = self._norm_a_pd_tf.logp(mean, mean)

        return a, logp
    
    def _eval_critic(self, s, g):
        s = np.reshape(s, [-1, self.get_state_size()])
        g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None

        critic_input = self._get_critic_inputs(s, g)
        val = self._critic_tf(critic_input).detach().numpy()

        return val

    def _record_flags(self):
        flags = int(0)
        if (self._exp_action):
            flags = flags | self.EXP_ACTION_FLAG
        return flags

    def _train_step(self):
        super()._train_step()

        critic_loss = self._update_critic()
        actor_loss = self._update_actor()
        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)

        critic_stepsize = self.critic_solver.get_stepsize()
        actor_stepsize = self.actor_solver.get_stepsize()
        
        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss) 
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)

        return

    def _update_critic(self):
        assert False
        # idx = self.replay_buffer.sample(self._local_mini_batch_size)
        # s = self.replay_buffer.get('states', idx)
        # g = self.replay_buffer.get('goals', idx) if self.has_goal() else None
        
        # tar_vals = self._calc_updated_vals(idx)
        # tar_vals = np.clip(tar_vals, self.val_min, self.val_max)
        
        # val_diff = self._tar_val_tf - self._critic_tf
        # self._critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(val_diff))

        # feed = {
        #     self._s_ph: s,
        #     self._g_ph: g,
        #     self._tar_val_ph: tar_vals
        # }

        # loss, grads = self.sess.run([self.critic_loss_tf, self.critic_grad_tf], feed)
        # self.critic_solver.update(grads)
        return loss
    
    def _update_actor(self):
        assert False

        if (self.actor_bound_loss_weight != 0.0):
            loss += self.actor_bound_loss_weight * self._build_action_bound_loss(self._norm_a_pd_tf, mean_actor)

        key = self.EXP_ACTION_FLAG
        idx = self.replay_buffer.sample_filtered(self._local_mini_batch_size, key)
        has_goal = self.has_goal()

        s = self.replay_buffer.get('states', idx)
        g = self.replay_buffer.get('goals', idx) if has_goal else None
        a = self.replay_buffer.get('actions', idx)

        V_new = self._calc_updated_vals(idx)
        V_old = self._eval_critic(s, g)
        adv = V_new - V_old

        feed = {
            self._s_ph: s,
            self._g_ph: g,
            self._a_ph: a,
            self._adv_ph: adv
        }

        loss, grads = self.sess.run([self._actor_loss_tf, self._actor_grad_tf], feed)
        self._actor_solver.update(grads)

        return loss

    def _calc_updated_vals(self, idx):
        r = self.replay_buffer.get('rewards', idx)

        if self.discount == 0:
            new_V = r
        else:
            next_idx = self.replay_buffer.get_next_idx(idx)
            s_next = self.replay_buffer.get('states', next_idx)
            g_next = self.replay_buffer.get('goals', next_idx) if self.has_goal() else None

            is_end = self.replay_buffer.is_path_end(idx)
            is_fail = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)
            is_succ = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)
            is_fail = np.logical_and(is_end, is_fail) 
            is_succ = np.logical_and(is_end, is_succ) 

            V_next = self._eval_critic(s_next, g_next)
            V_next[is_fail] = self.val_fail
            V_next[is_succ] = self.val_succ

            new_V = r + self.discount * V_next
        return new_V

    def _log_val(self, s, g):
        #TODO: ignoring
        # val = self._eval_critic(s, g)
        # norm_val = (1.0 - self.discount) * val
        # self.world.env.log_val(self.id, norm_val[0])
        return

    def _build_replay_buffer(self, buffer_size):
        super()._build_replay_buffer(buffer_size)
        self.replay_buffer.add_filter_key(self.EXP_ACTION_FLAG)
        return

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, init_output_scale, noise, activation = nn.ReLU) -> None:
        super(Actor, self).__init__()

        layers = [input_dim] + hidden_layers
        module = []
        for i in range(len(layers)-1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            module.append(layer)
            module.append(activation())
        self.hidden = nn.Sequential(*module)

        self.dim = output_dim
        #mean init use random uniform init minval=-init_output_scale, maxval=init_output_scale
        #mean bias init use zeros
        #logstd_kernel init use random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale)
        #logstd bias init np.log(self.exp_params_curr.noise) * np.ones(a_size) logstd_bias_init.astype(np.float32)

        #build parameter
        #mean
        self._mean = nn.Linear(hidden_layers[-1], self.dim)
        nn.init.uniform_(self._mean.weight, -init_output_scale, init_output_scale)
        nn.init.zeros_(self._mean.bias)
        #logstd
        logstd_bias = np.log(noise) * np.ones(output_dim)
        logstd_bias = logstd_bias.astype(np.float32)
        self.logstd = torch.FloatTensor(logstd_bias)
        self.std = torch.exp(self.logstd)

    def forward(self, x):
        assert False
        # x = self.fc(x)
        # return x

    def mean(self, x):
        x = self.hidden(x)
        mean = self._mean(x)
        return mean

    def sample(self, mean):
        noise = torch.rand_like(mean)
        samples = self.std * noise + mean
        return samples

    def logp(self, x, mean):
        diff_tf = x - mean
        logp_tf = -0.5 * torch.sum(torch.square(diff_tf / self.std), dim=-1)
        logp_tf += -0.5 * self.dim * np.log(2.0 * np.pi) - torch.sum(self.logstd, dim=-1)
        return logp_tf

    def flat_params(self, mean):
        mean_tf = mean
        logstd_tf = self.logstd
        mean_shape = list(mean_tf.shape)
        logstd_shape = list(logstd_tf.shape)
        
        if (len(mean_shape) == 2 and len(logstd_shape) == 1):
            mean_rows = mean_tf.shape[0]
            logstd_tf = torch.reshape(logstd_tf, [1, logstd_shape[-1]])
            logstd_tf = logstd_tf.repeat([mean_rows, 1])
        else:
            assert (len(mean_shape) == len(logstd_shape))

        params = torch.concat([mean_tf, logstd_tf], axis=-1)
        return params

    def param_bound_loss(self, bound_min, bound_max, mean):
        flat_params = self.flat_params(mean)
        num_params = list(flat_params.shape)[-1]
        assert(bound_min.shape[-1] == num_params)
        assert(bound_max.shape[-1] == num_params)

        bound_min = torch.FloatTensor(bound_min)
        bound_max = torch.FloatTensor(bound_max)

        violation_min = torch.minimum(flat_params-bound_min, torch.zeros_like(bound_min))
        violation_max = torch.maximum(flat_params-bound_max, torch.zeros_like(bound_max))
        violation = torch.sum(torch.square(violation_min), axis=-1) \
                    + torch.sum(torch.square(violation_max), axis=-1)
        loss = 0.5 * torch.mean(violation)

        return loss

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation = nn.ReLU) -> None:
        super(Critic, self).__init__()

        layers = [input_dim] + hidden_layers
        module = []
        for i in range(len(layers)-1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            module.append(layer)
            module.append(activation())
        self.hidden = nn.Sequential(*module)
        self.fc = nn.Linear(hidden_layers[-1], output_dim)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.hidden(x)
        x = self.fc(x)
        x = torch.squeeze(x, -1)
        return x
