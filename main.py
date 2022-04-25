import argparse
import math
import os
from datetime import datetime

import torch
import env as envpy
from ppo.PPO import PPO
from discriminator import Discriminator

class LoadArgFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def train_loop():
    # while True:
        #update world
        #check valid epi(whether humanoid lin vel and ang vel < 100)
        #if valid
        #   if episode ended
        #       do training
        #       reset world
        #else
        #   reset world
    pass

if __name__ == '__main__':
    #parse argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--arg_file', type=open, action=LoadArgFromFile, help='argment file path')
    arg_parser.add_argument('--motion_file', type=str, default='./motion_file/sfu_walking.txt', help='motion file path')
    arg_parser.add_argument('--draw', action=argparse.BooleanOptionalAction, default=False, help='render the environment')
    arg_parser.add_argument('--timestep', type=float, default=1/240, help='simulation time step')
    arg_parser.add_argument('--fall_contact_bodies', type=int, nargs="+")
    args = arg_parser.parse_args()
    
    print(args)

    env = envpy.Env(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ####### initialize environment hyperparameters ######
    env_name = "sfu_walking"

    has_continuous_action_space = True  # continuous action space; else discrete

    print_freq = 10         # print avg reward in the interval (in num iter)
    log_freq = 2            # log avg reward in the interval (in num iter)
    save_model_freq = 1     # save model frequency (in num iter)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    # update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # state space dimension
    state_dim = env.GetStateDim()

    # action space dimension
    action_dim = env.GetActionDim()

    disc_input_dim = env.GetDiscInputDim()
    print(disc_input_dim)

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # initialize a discriminator
    disc = Discriminator(disc_input_dim, [1024, 512]).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr = 0.00001, weight_decay=0.0005)

    disc_agent_replay_buffer = []
    disc_expert_replay_buffer = []

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    num_sample = 0

    # training loop
    while True:

        env.reset()
        state = env.RecordAgentObs()
        current_ep_reward = 0

        done = False
        while not done:
            disc_agent_frame, disc_agent_vel = env.RecordAgentFrameAndVel()
            action = ppo_agent.select_action(state)
            env.step(action)
            next_state = env.RecordAgentObs()
            disc_agent_next_frame, disc_agent_next_vel = env.RecordAgentFrameAndVel()
            disc_agent_obs = env.RecordAgentDiscObs(disc_agent_frame, disc_agent_vel, disc_agent_next_frame, disc_agent_next_vel)
            disc_agent_replay_buffer.append(disc_agent_obs)
            disc_expert_obs = env.RecordExpertDiscObs()
            disc_expert_replay_buffer.append(disc_expert_obs)
            
            done = env.terminate()
            ppo_agent.buffer.is_terminals.append(done)

            state = next_state
            num_sample += 1
        
        #train discriminator
        expert_replay_buffer_len = len(disc_expert_replay_buffer)
        if expert_replay_buffer_len >= 256:
            num_update = math.floor(expert_replay_buffer_len / 256)
            for i in range(num_update):
                start_idx = 256*i
                end_idx = 256*(i+1)
            disc_expert_return = disc(torch.FloatTensor(disc_expert_replay_buffer[start_idx:end_idx]).to(device))
            disc_agent_return = disc(torch.FloatTensor(disc_agent_replay_buffer[start_idx:end_idx]).to(device))
            disc_loss_expert = (0.5 * (torch.square(disc_expert_return - 1)).sum()).mean()
            disc_loss_agent =  (0.5 * (torch.square(disc_agent_return + 1)).sum()).mean()
            disc_loss = 0.5 * (disc_loss_expert + disc_loss_agent)
            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()
        
        if len(ppo_agent.buffer.actions) >=256:
            rewards = 1.0 - 0.25 * torch.square(1.0 - disc(torch.FloatTensor(disc_agent_replay_buffer).to(device)))
            rewards = torch.maximum(rewards, torch.zeros_like(rewards)).reshape(1,-1).detach().cpu().numpy()[0].tolist()
            ppo_agent.buffer.rewards += rewards
            ppo_agent.update()
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)    #TODO: not sure if decay everytime is okay

            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("Num of Sample:", num_sample)
            print("--------------------------------------------------------------------------------------------")

            del disc_expert_replay_buffer
            del disc_agent_replay_buffer
            disc_expert_replay_buffer = []
            disc_agent_replay_buffer = []