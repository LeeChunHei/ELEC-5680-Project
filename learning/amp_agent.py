import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import Env
# import learning.nets.net_builder as net_builder
from learning.ppo_agent import PPOAgent
# import learning.solvers.mpi_solver as mpi_solver
from learning.tf_normalizer import TFNormalizer
from learning.replay_buffer_rand_storage import ReplayBufferRandStorage

'''
Adversarial Motion Prior Agent
'''

class AMPAgent(PPOAgent):
    NAME = "AMP"
    
    TASK_REWARD_LERP_KEY = "TaskRewardLerp"

    DISC_NET_KEY = "DiscNet"
    DISC_INIT_OUTPUT_SCALE_KEY = "DiscInitOutputScale"
    DISC_WEIGHT_DECAY_KEY = "DiscWeightDecay"
    DISC_LOGIT_REG_WEIGHT_KEY = "DiscLogitRegWeight"
    DISC_STEPSIZE_KEY = "DiscStepSize"
    DISC_MOMENTUM_KEY  = "DiscMomentum"
    DISC_BATCH_SIZE_KEY = "DiscBatchSize"
    DISC_STEPS_PER_BATCH_KEY = "DiscStepsPerBatch"
    DISC_EXPERT_BUFFER_SIZE_KEY = "DiscExpertBufferSize"
    DISC_AGENT_BUFFER_SIZE_KEY = "DiscAgentBufferSize"

    REWARD_SCALE_KEY = "RewardScale"
    DISC_GRAD_PENALTY_KEY = "DiscGradPenalty"

    DISC_LOGIT_NAME = "disc_logits"
    DISC_SCOPE = "disc"
    
    def __init__(self, json_data, env, writer):
        super().__init__(json_data, env, writer)

        self._disc_reward_mean = 0.0
        self._disc_reward_std = 0.0
        self._reward_min = np.inf
        self._reward_max = -np.inf
        
        self._build_disc_replay_buffer()

        return

    def __str__(self):
        info_str = super().__str__()
        info_str = info_str[:-2] + ',\n "AMPObsDim": "{:d}"'.format(self._get_amp_obs_size()) + info_str[-2:]
        return info_str

    def _load_params(self, json_data):
        super()._load_params(json_data)

        self._task_reward_lerp = 0.5 if (self.TASK_REWARD_LERP_KEY not in json_data) else json_data[self.TASK_REWARD_LERP_KEY]

        self._disc_batchsize = int(256) if (self.DISC_BATCH_SIZE_KEY not in json_data) else int(json_data[self.DISC_BATCH_SIZE_KEY])
        self._disc_steps_per_batch = int(1) if (self.DISC_STEPS_PER_BATCH_KEY not in json_data) else int(json_data[self.DISC_STEPS_PER_BATCH_KEY])
        self._disc_expert_buffer_size = int(100000) if (self.DISC_EXPERT_BUFFER_SIZE_KEY not in json_data) else int(json_data[self.DISC_EXPERT_BUFFER_SIZE_KEY])
        self._disc_agent_buffer_size = int(100000) if (self.DISC_AGENT_BUFFER_SIZE_KEY not in json_data) else int(json_data[self.DISC_AGENT_BUFFER_SIZE_KEY])
        
        return

    def save_model(self, out_path):
        super().save_model(out_path)
        #TODO: need to see what to save, and need to be called by upper class        
        checkpoint = {}
        checkpoint['disc'] = self._disc_logits_tf .state_dict()
        checkpoint['disc_opt'] = self.disc_opt.state_dict()
        torch.save(checkpoint, out_path+'_amp_agent')

    def load_model(self, in_path):
        super().save_model(in_path)
        checkpoint = torch.load(in_path+"_amp_agent")
        self._disc_logits_tf.load_state_dict(checkpoint["disc"])
        self.disc_opt.load_state_dict(checkpoint['disc_opt'])

    def _build_disc_replay_buffer(self):
        num_procs = 1#mpi_util.get_num_procs()
        local_disc_expert_buffer_size = int(np.ceil(self._disc_expert_buffer_size / num_procs))
        self._disc_expert_buffer = ReplayBufferRandStorage(local_disc_expert_buffer_size)
        
        local_disc_agent_buffer_size = int(np.ceil(self._disc_agent_buffer_size / num_procs))
        self._disc_agent_buffer = ReplayBufferRandStorage(local_disc_agent_buffer_size)
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        self._amp_obs_norm = TFNormalizer(self._get_amp_obs_size(), self._get_amp_obs_norm_group())
        self._amp_obs_norm.set_mean_std(-self._get_amp_obs_offset(), 1 / self._get_amp_obs_scale())
        return

    def _load_normalizers(self):
        super()._load_normalizers()
        self._amp_obs_norm.load()
        return
    
    def _update_normalizers(self):
        super()._update_normalizers()
        self._amp_obs_norm.update()
        return
    
    def _sync_solvers(self):
        # super()._sync_solvers()
        # self._disc_solver.sync()
        return

    def _build_nets(self, json_data):
        super()._build_nets(json_data)

        assert self.DISC_NET_KEY in json_data

        disc_net_name = json_data[self.DISC_NET_KEY]
        disc_init_output_scale = 1 if (self.DISC_INIT_OUTPUT_SCALE_KEY not in json_data) else json_data[self.DISC_INIT_OUTPUT_SCALE_KEY]
        self._reward_scale = 1.0 if (self.REWARD_SCALE_KEY not in json_data) else json_data[self.REWARD_SCALE_KEY]
        
        amp_obs_size = self._get_amp_obs_size()

        # setup input tensors
        # self._amp_obs_expert_ph = tf.placeholder(tf.float32, shape=[None, amp_obs_size], name="amp_obs_expert")
        # self._amp_obs_agent_ph = tf.placeholder(tf.float32, shape=[None, amp_obs_size], name="amp_obs_agent")
        
        # self._disc_expert_inputs = self._get_disc_expert_inputs()
        # self._disc_agent_inputs = self._get_disc_agent_inputs()

        self._disc_logits_tf = self._build_disc_net(disc_net_name, amp_obs_size, disc_init_output_scale)
        # self._disc_logits_agent_tf = self._build_disc_net(disc_net_name, self._disc_agent_inputs, disc_init_output_scale, reuse=True)
                
        # if (self._disc_logits_expert_tf != None):
        #     print("Built discriminator net: " + disc_net_name)

        # self._disc_prob_agent_tf = tf.sigmoid(self._disc_logits_agent_tf)
        # self._abs_logit_agent_tf = tf.reduce_mean(tf.abs(self._disc_logits_agent_tf))
        # self._avg_prob_agent_tf = tf.reduce_mean(self._disc_prob_agent_tf)

        return

    def _build_losses(self, json_data):
        super()._build_losses(json_data)

        # disc_weight_decay = 0 if (self.DISC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.DISC_WEIGHT_DECAY_KEY]
        # disc_logit_reg_weight = 0 if (self.DISC_LOGIT_REG_WEIGHT_KEY not in json_data) else json_data[self.DISC_LOGIT_REG_WEIGHT_KEY]
        # disc_grad_penalty = 0.0 if (self.DISC_GRAD_PENALTY_KEY not in json_data) else json_data[self.DISC_GRAD_PENALTY_KEY]
        
        # disc_loss_expert_tf = self.build_disc_loss_pos(self._disc_logits_expert_tf)
        # disc_loss_agent_tf = self.build_disc_loss_neg(self._disc_logits_agent_tf)
        # disc_loss_expert_tf = tf.reduce_mean(disc_loss_expert_tf)
        # disc_loss_agent_tf = tf.reduce_mean(disc_loss_agent_tf)

        # self._disc_loss_tf = 0.5 * (disc_loss_agent_tf + disc_loss_expert_tf)

        # self._acc_expert_tf = tf.reduce_mean(tf.cast(tf.greater(self._disc_logits_expert_tf, 0), tf.float32))
        # self._acc_agent_tf = tf.reduce_mean(tf.cast(tf.less(self._disc_logits_agent_tf, 0), tf.float32))
        
        # if (disc_weight_decay != 0):
        #     self._disc_loss_tf += disc_weight_decay * self._weight_decay_loss(self.MAIN_SCOPE + "/" + self.DISC_SCOPE)
                    
        # #TODO: skipping logit reg loss and grad penalty loss here
        # if (disc_logit_reg_weight != 0):
        #     self._disc_loss_tf += disc_logit_reg_weight * self._disc_logit_reg_loss()
        
        # if (disc_grad_penalty != 0):
        #     self._grad_penalty_loss_tf = self._disc_grad_penalty_loss(in_tfs=self._disc_expert_inputs, out_tf=self._disc_logits_expert_tf)
        #     self._disc_loss_tf += disc_grad_penalty * self._grad_penalty_loss_tf
        # else:
        #     self._grad_penalty_loss_tf = tf.constant(0.0, dtype=tf.float32)

        return

    def _build_solvers(self, json_data):
        super()._build_solvers(json_data)

        disc_stepsize = 0.001 if (self.DISC_STEPSIZE_KEY not in json_data) else json_data[self.DISC_STEPSIZE_KEY]
        disc_momentum = 0.9 if (self.DISC_MOMENTUM_KEY not in json_data) else json_data[self.DISC_MOMENTUM_KEY]
        disc_weight_decay = 0 if (self.DISC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.DISC_WEIGHT_DECAY_KEY]
        
        # disc_vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.DISC_SCOPE)
        self.disc_opt = torch.optim.SGD(self._disc_logits_tf.parameters(), lr=disc_stepsize, momentum=disc_momentum, weight_decay=disc_weight_decay)
        # self._disc_grad_tf = tf.gradients(self._disc_loss_tf, disc_vars)
        # self._disc_solver = mpi_solver.MPISolver(self.sess, disc_opt, disc_vars)
        
        return

    def _build_disc_net(self, net_name, input_dim, init_output_scale, reuse=False):
        out_size = 1
        if net_name == "fc_2layers_1024units":
            logits_tf = Discriminator(input_dim, out_size, [1024, 512], init_output_scale)
        else:
            assert False, net_name+" not supported"
        return logits_tf

    def _get_disc_expert_inputs(self, amp_obs_expert):
        norm_obs_tf = self._amp_obs_norm.normalize_tf(amp_obs_expert)
        input_tfs = norm_obs_tf
        return input_tfs
       
    def _get_disc_agent_inputs(self, amp_obs_agent):
        norm_obs_tf = self._amp_obs_norm.normalize_tf(amp_obs_agent)
        input_tfs = norm_obs_tf
        return input_tfs
 
    def _disc_logit_reg_loss(self):
        #TODO: don't know how to implement in pytorch
        pass
        # vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.DISC_SCOPE)
        # logit_vars = [v for v in vars if (self.DISC_LOGIT_NAME in v.name and "bias" not in v.name)]
        # loss_tf = tf.add_n([tf.nn.l2_loss(v) for v in logit_vars])
        # return loss_tf

    def _disc_grad_penalty_loss(self, in_tfs, out_tf):
        #TODO: don't know how to implement in pytorch
        pass
        # grad_tfs = tf.gradients(ys=out_tf, xs=in_tfs)
        # grad_tf = tf.concat(grad_tfs, axis=-1)
        # norm_tf = tf.reduce_sum(tf.square(grad_tf), axis=-1)
        # loss_tf = 0.5 * tf.reduce_mean(norm_tf)
        # return loss_tf

    def reset(self):
        super().reset()

        self.path.amp_obs_expert = []
        self.path.amp_obs_agent = []
        return

    def _store_path(self, path):
        path_id = super()._store_path(path)

        valid_path = (path_id != -1)
        if (valid_path):
            disc_expert_path_id = self._disc_expert_buffer.store(path.amp_obs_expert)
            assert(disc_expert_path_id != -1)

            disc_agent_path_id = self._disc_agent_buffer.store(path.amp_obs_agent)
            assert(disc_agent_path_id != -1)

        return path_id
    
    def _update_new_action(self):
        first_step = self._is_first_step()

        super()._update_new_action()

        if (not first_step):
            self._record_amp_obs()

        return

    def _end_path(self):
        super()._end_path()
        self._record_amp_obs()
        return

    def _record_amp_obs(self):
        obs_expert = self._record_amp_obs_expert()
        obs_agent = self._record_amp_obs_agent()
        self.path.amp_obs_expert.append(obs_expert)
        self.path.amp_obs_agent.append(obs_agent)
        return
    
    def _enable_amp_task_reward(self):
        enable = False#self.world.env.enable_amp_task_reward()
        return enable

    def _get_amp_obs_size(self):
        amp_obs_size = self.env.GetDiscInputDim()
        return amp_obs_size

    def _get_amp_obs_offset(self):
        offset = np.array(self.env.build_amp_obs_offset(0))
        return offset
    
    def _get_amp_obs_scale(self):
        offset = np.array(self.env.build_amp_obs_scale(0))
        return offset
    
    def _get_amp_obs_norm_group(self):
        norm_group = np.array(self.env.build_amp_obs_norm_groups(0), dtype=np.int32)
        return norm_group
    
    def _record_amp_obs_expert(self):
        obs_expert = np.array(self.env.record_amp_obs_expert(0))
        return obs_expert

    def _record_amp_obs_agent(self):
        obs_agent = np.array(self.env.record_amp_obs_agent(0))
        return obs_agent

    def _record_normalizers(self, path):
        super()._record_normalizers(path)

        self._amp_obs_norm.record(np.array(path.amp_obs_expert))
        self._amp_obs_norm.record(np.array(path.amp_obs_agent))
        return

    def _logits_to_reward(self, logits):
        r = 1.0 - 0.25 * np.square(1.0 - logits)
        r = np.maximum(r, 0.0)
        return r

    def _train_step(self):
        disc_info = self._update_disc()

        disc_info["reward_mean"] = self._disc_reward_mean
        disc_info["reward_std"] = self._disc_reward_std
        disc_info = disc_info#mpi_util.reduce_dict_mean(disc_info)
        
        super()._train_step()
        
        print("#################################")
        print("Disc_Loss", disc_info["loss"])
        print("Disc_Acc_Expert", disc_info["acc_expert"])
        print("Disc_Acc_Agent", disc_info["acc_agent"])
        print("Disc_Stepsize", self.get_disc_stepsize())
        print("Disc_Steps", self.get_disc_steps())
        print("Disc_Prob", disc_info["prob_agent"])
        print("Disc_Abs_Logit", disc_info["abs_logit"])
        print("Disc_Reward_Mean", disc_info["reward_mean"])
        print("Disc_Reward_Std", disc_info["reward_std"])
        print("#################################")

        if (self._enable_grad_penalty()):
            print("Grad_Penalty", disc_info["grad_penalty"])

        return

    def _update_disc(self):
        info = None

        num_procs = 1#mpi_util.get_num_procs()
        local_expert_batch_size = int(np.ceil(self._disc_batchsize / num_procs))
        local_agent_batch_size = local_expert_batch_size
        
        steps_per_batch = self._disc_steps_per_batch
        local_sample_count = self.replay_buffer.get_current_size()
        global_sample_count = local_sample_count#int(mpi_util.reduce_sum(local_sample_count))
        
        num_steps = int(np.ceil(steps_per_batch * global_sample_count / (num_procs * local_expert_batch_size)))

        for b in range(num_steps):
            disc_expert_batch = self._disc_expert_buffer.sample(local_expert_batch_size)
            obs_expert = self._disc_expert_buffer.get(disc_expert_batch)
            
            disc_agent_batch = self._disc_agent_buffer.sample(local_agent_batch_size)
            obs_agent = self._disc_agent_buffer.get(disc_agent_batch)

            curr_info = self._step_disc(obs_expert=obs_expert, obs_agent=obs_agent)

            if (info is None):
                info = curr_info
            else:
                for k, v in curr_info.items():
                    info[k] += v
        
        for k in info.keys():
            info[k] /= num_steps

        return info
        
    def _step_disc(self, obs_expert, obs_agent):
        expert_obs = self._get_disc_expert_inputs(obs_expert)
        agent_obs = self._get_disc_agent_inputs(obs_agent)

        disc_logits_expert = self._disc_logits_tf(expert_obs)
        disc_logits_agent = self._disc_logits_tf(agent_obs)

        disc_loss_expert = torch.mean(0.5 * torch.sum(torch.square(disc_logits_expert - 1), axis=-1))
        disc_loss_agent = torch.mean(0.5 * torch.sum(torch.square(disc_logits_agent + 1), axis=-1))

        disc_loss_tf = 0.5 * ((disc_loss_agent+disc_loss_expert))
        self.disc_opt.zero_grad()
        disc_loss_tf.backward()
        self.disc_opt.step()

        acc_expert_tf = torch.mean(torch.greater(disc_logits_expert, 0).float())
        acc_agent_tf = torch.mean(torch.less(disc_logits_agent, 0).float())

        disc_prob_agent_tf = torch.sigmoid(disc_logits_agent)
        abs_logit_agent_tf = torch.mean(torch.abs(disc_logits_agent))
        avg_prob_agent_tf = torch.mean(disc_prob_agent_tf)

        info = {
            "loss": disc_loss_tf,
            "acc_expert": acc_expert_tf,
            "acc_agent": acc_agent_tf,
            "prob_agent": avg_prob_agent_tf,
            "abs_logit": abs_logit_agent_tf,
            "grad_penalty": 0,#"didn't implement",
        }
        return info
    
    def get_disc_stepsize(self):
        return "i don't know"
        # return self._disc_solver.get_stepsize()
    
    def get_disc_steps(self):
        return "i don't know"
        # return self._disc_solver.iter

    def _enable_grad_penalty(self):
        return False#self._grad_penalty_loss_tf.op.type != "Const"
    
    def _fetch_batch_rewards(self, start_idx, end_idx):
        idx = np.array(list(range(start_idx, end_idx)))  
        rewards = self._batch_calc_reward(idx)
        return rewards

    def _batch_calc_reward(self, idx):
        obs_agent = self.replay_buffer.get("amp_obs_agent", idx)
        disc_r, _ = self._calc_disc_reward(obs_agent)
        
        end_mask = self.replay_buffer.is_path_end(idx)
        valid_mask = np.logical_not(end_mask)

        disc_r *= self._reward_scale
        valid_disc_r = disc_r[valid_mask]
        self._disc_reward_mean = np.mean(valid_disc_r)
        self._disc_reward_std = np.std(valid_disc_r)
        
        if (self._enable_amp_task_reward()):
            task_r = self.replay_buffer.get("rewards", idx)
            r = self._lerp_reward(disc_r, task_r)
        else:
            r = disc_r
        
        curr_reward_min = np.amin(r)
        curr_reward_max = np.amax(r)
        self._reward_min = np.minimum(self._reward_min, curr_reward_min)
        self._reward_max = np.maximum(self._reward_max, curr_reward_max)
        reward_data = np.array([self._reward_min, -self._reward_max])
        reward_data = reward_data#mpi_util.reduce_min(reward_data)

        self._reward_min = reward_data[0]
        self._reward_max = -reward_data[1]

        return r

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self._task_reward_lerp) * disc_r + self._task_reward_lerp * task_r
        return r

    def _calc_disc_reward(self, amp_obs):
        agent_obs = self._get_disc_agent_inputs(amp_obs)

        logits = self._disc_logits_tf(agent_obs).detach().numpy()

        r = self._logits_to_reward(logits)
        r = r[:, 0]
        return r, logits

    def _compute_batch_vals(self, start_idx, end_idx):
        states = self.replay_buffer.get_all("states")[start_idx:end_idx]
        goals = self.replay_buffer.get_all("goals")[start_idx:end_idx] if self.has_goal() else None
        # print('eval', states, goals)
        vals = self._eval_critic(states, goals)
        # print('eval val', vals)

        val_min = self._reward_min / (1.0 - self.discount)
        val_max = self._reward_max / (1.0 - self.discount)
        vals = np.clip(vals, val_min, val_max)
        print("valminmax", val_min, val_max, vals)
        
        idx = np.array(list(range(start_idx, end_idx)))        
        is_end = self.replay_buffer.is_path_end(idx)
        is_fail = self.replay_buffer.check_terminal_flag(idx, 1)
        is_succ = self.replay_buffer.check_terminal_flag(idx, 2)
        is_fail = np.logical_and(is_end, is_fail) 
        is_succ = np.logical_and(is_end, is_succ)
        vals[is_fail] = self.val_fail
        vals[is_succ] = self.val_succ

        return vals


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, init_output_scale, activation = nn.ReLU) -> None:
        super(Discriminator, self).__init__()

        layers = [input_dim] + hidden_layers
        module = []
        for i in range(len(layers)-1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            module.append(layer)
            module.append(activation())
        self.hidden = nn.Sequential(*module)

        self.fc = nn.Linear(hidden_layers[-1], output_dim)
        nn.init.uniform_(self.fc.weight, -init_output_scale, init_output_scale)

    def forward(self, x):
        x = self.hidden(x)
        x = self.fc(x)
        return x
