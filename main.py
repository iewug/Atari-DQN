import gymnasium as gym
import argparse
from model import DQN, DuelDQN
from torch import optim
from utils import Transition, ReplayMemory, VideoRecorder
from wrapper import AtariWrapper
import numpy as np
import random
import torch
import torch.nn as nn
from itertools import count
import os
import matplotlib.pyplot as plt
import math
from collections import deque

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--env-name',default="breakout",type=str,choices=["pong","breakout","boxing"], help="env name")
parser.add_argument('--model', default="dqn", type=str, choices=["dqn","dueldqn"], help="dqn model")
parser.add_argument('--gpu',default=0,type=int,help="which gpu to use")
parser.add_argument('--lr', default=2.5e-4, type=float, help="learning rate")
parser.add_argument('--epoch', default=10001, type=int, help="training epoch")
parser.add_argument('--batch-size', default=32, type=int, help="batch size")
parser.add_argument('--ddqn',action='store_true', help="double dqn/dueldqn")
parser.add_argument('--eval-cycle', default=500, type=int, help="evaluation cycle")
args = parser.parse_args()

# some hyperparameters
GAMMA = 0.99 # bellman function
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 50000
WARMUP = 1000 # don't update net until WARMUP steps

steps_done = 0
eps_threshold = EPS_START
def select_action(state:torch.Tensor)->torch.Tensor:
    '''
    epsilon greedy
    - epsilon: choose random action
    - 1-epsilon: argmax Q(a,s)

    Input: state shape (1,4,84,84)

    Output: action shape (1,1)
    '''
    global eps_threshold
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]]).to(args.gpu)


# environment
if args.env_name == "pong":
    env = gym.make("PongNoFrameskip-v4")
elif args.env_name == "breakout":
    env = gym.make("BreakoutNoFrameskip-v4")
else:
    env = gym.make("BoxingNoFrameskip-v4")
env = AtariWrapper(env)

n_action = env.action_space.n # pong:6; breakout:4; boxing:18

# make dir to store result
if args.ddqn:
    methodname = f"double_{args.model}"
else:
    methodname = args.model
log_dir = os.path.join(f"log_{args.env_name}",methodname)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir,"log.txt")


# video
video = VideoRecorder(log_dir)

# create network and target network
if args.model == "dqn":
    policy_net = DQN(in_channels=4, n_actions=n_action).to(args.gpu)
    target_net = DQN(in_channels=4, n_actions=n_action).to(args.gpu)
else:
    policy_net = DuelDQN(in_channels=4, n_actions=n_action).to(args.gpu)
    target_net = DuelDQN(in_channels=4, n_actions=n_action).to(args.gpu)
# let target model = model
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# replay memory
memory = ReplayMemory(50000)

# optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)

# warming up
print("Warming up...")
warmupstep = 0
for epoch in count():
    obs, info = env.reset() # (84,84)
    obs = torch.from_numpy(obs).to(args.gpu) #(84,84)
    # stack four frames together, hoping to learn temporal info
    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0) #(1,4,84,84)

    # step loop
    for step in count():
        warmupstep += 1
        # take one step
        action = torch.tensor([[env.action_space.sample()]]).to(args.gpu)
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # convert to tensor
        reward = torch.tensor([reward],device=args.gpu) # (1)
        done = torch.tensor([done],device=args.gpu) # (1)
        next_obs = torch.from_numpy(next_obs).to(args.gpu) # (84,84)
        next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
        
        # store the transition in memory
        memory.push(obs,action,next_obs,reward,done)
        
        # move to next state
        obs = next_obs

        if done:
            break

    if warmupstep > WARMUP:
        break

rewardList = []
lossList = []
rewarddeq = deque([], maxlen=100)
lossdeq = deque([],maxlen=100)
avgrewardlist = []
avglosslist = []
# epoch loop 
for epoch in range(args.epoch):
    obs, info = env.reset() # (84,84)
    obs = torch.from_numpy(obs).to(args.gpu) #(84,84)
    # stack four frames together, hoping to learn temporal info
    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0) #(1,4,84,84)

    total_loss = 0.0
    total_reward = 0

    # step loop
    for step in count():
        # take one step
        action = select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        
        # convert to tensor
        reward = torch.tensor([reward],device=args.gpu) # (1)
        done = torch.tensor([done],device=args.gpu) # (1)
        next_obs = torch.from_numpy(next_obs).to(args.gpu) # (84,84)
        next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
        
        # store the transition in memory
        memory.push(obs,action,next_obs,reward,done)
        
        # move to next state
        obs = next_obs

        # train
        policy_net.train()
        transitions = memory.sample(args.batch_size)
        batch = Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.
        state_batch = torch.cat(batch.state) # (bs,4,84,84)
        next_state_batch = torch.cat(batch.next_state) # (bs,4,84,84)
        action_batch = torch.cat(batch.action) # (bs,1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1) # (bs,1)
        done_batch = torch.cat(batch.done).unsqueeze(1) #(bs,1)

        # Q(st,a)
        state_qvalues = policy_net(state_batch) # (bs,n_actions)
        selected_state_qvalue = state_qvalues.gather(1,action_batch) # (bs,1)
        
        with torch.no_grad():
            # Q'(st+1,a)
            next_state_target_qvalues = target_net(next_state_batch) # (bs,n_actions)
            if args.ddqn:
                # Q(st+1,a)
                next_state_qvalues = policy_net(next_state_batch) # (bs,n_actions)
                # argmax Q(st+1,a)
                next_state_selected_action = next_state_qvalues.max(1,keepdim=True)[1] # (bs,1)
                # Q'(st+1,argmax_a Q(st+1,a))
                next_state_selected_qvalue = next_state_target_qvalues.gather(1,next_state_selected_action) # (bs,1)
            else:
                # max_a Q'(st+1,a)
                next_state_selected_qvalue = next_state_target_qvalues.max(1,keepdim=True)[0] # (bs,1)

        # td target
        tdtarget = next_state_selected_qvalue * GAMMA * ~done_batch + reward_batch # (bs,1)

        # optimize
        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_state_qvalue, tdtarget)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # let target_net = policy_net every 1000 steps
        if steps_done % 1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            # eval
            if epoch % args.eval_cycle == 0:
                with torch.no_grad():
                    video.reset()
                    if args.env_name == "pong":
                        evalenv = gym.make("PongNoFrameskip-v4")
                    elif args.env_name == "breakout":
                        evalenv = gym.make("BreakoutNoFrameskip-v4")
                    else:
                        evalenv = gym.make("BoxingNoFrameskip-v4")
                    evalenv = AtariWrapper(evalenv,video=video)
                    obs, info = evalenv.reset()
                    obs = torch.from_numpy(obs).to(args.gpu)
                    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)
                    evalreward = 0
                    policy_net.eval()
                    for _ in count():
                        action = policy_net(obs).max(1)[1]
                        next_obs, reward, terminated, truncated, info = evalenv.step(action.item())
                        evalreward += reward
                        next_obs = torch.from_numpy(next_obs).to(args.gpu) # (84,84)
                        next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
                        obs = next_obs
                        if terminated or truncated:
                            if info["lives"] == 0: # real end
                                break
                            else:
                                obs, info = evalenv.reset()
                                obs = torch.from_numpy(obs).to(args.gpu)
                                obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)
                    evalenv.close()
                    video.save(f"{epoch}.mp4")
                    torch.save(policy_net, os.path.join(log_dir,f'model{epoch}.pth')) 
                    print(f"Eval epoch {epoch}: Reward {evalreward}")
            break
    
    rewardList.append(total_reward)
    lossList.append(total_loss)
    rewarddeq.append(total_reward)
    lossdeq.append(total_loss)
    avgreward = sum(rewarddeq)/len(rewarddeq)
    avgloss = sum(lossdeq)/len(lossdeq)
    avglosslist.append(avgloss)
    avgrewardlist.append(avgreward)

    output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {eps_threshold:.2f}, TotalStep {steps_done}"
    print(output)
    with open(log_path,"a") as f:
        f.write(f"{output}\n")

env.close()


# plot loss-epoch and reward-epoch
plt.figure(1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(range(len(lossList)),lossList,label="loss")
plt.plot(range(len(lossList)),avglosslist,label="avg")
plt.legend()
plt.savefig(os.path.join(log_dir,"loss.png"))

plt.figure(2)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.plot(range(len(rewardList)),rewardList,label="reward")
plt.plot(range(len(rewardList)),avgrewardlist, label="avg")
plt.legend()
plt.savefig(os.path.join(log_dir,"reward.png"))