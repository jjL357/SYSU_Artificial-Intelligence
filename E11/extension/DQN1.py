# original implementation

import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import random
from collections import deque


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 添加一个全连接层

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 计算输出
        # print(x.shape) learn() is [128, 2] ; choose action is [2]
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def len(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size) 
        # can do priority Reply buffer
        obs, actions, rewards, next_obs, dones = zip(*transitions)
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs), np.array(dones)

    def clean(self):
        self.buffer.clear()


class DQN:
    def __init__(self, env, input_size, hidden_size, output_size): # 初始化参数
        self.env = env
        self.eval_net = QNet(input_size, hidden_size, output_size)
        self.target_net = QNet(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps = args.eps
        self.gamma = args.gamma
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0

    def choose_action(self, obs):
        if np.random.uniform(0, 1) < self.eps: # 0.1 概率随机选择
            action = self.env.action_space.sample()  # 随机选择动作
        else:
            with torch.no_grad():
                q_values = self.eval_net(obs) # handle obs with forward function
                action = q_values.argmax().item()  # 选择Q值最大的动作
        return action

    def store_transition(self, *transition):
        self.buffer.push(*transition) # 5个参数 obs, action, reward, next_obs, done

    def learn(self): # 核心，参数更新方法
        if self.buffer.len() < args.batch_size: # default 128
            return

        if self.learn_step % args.update_target == 0: # df 100
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards) 
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)

        # what is the return of net? batch size * action number 
        # what is gather()? https://zhuanlan.zhihu.com/p/352877584
        # dqn: Q(st, at) <-> rt + max a Q'(st+1, a)
        # Qpi(s, a)
        # print("actions is:")
        # print(actions.shape)
        temp = self.eval_net(obs)
        # print("temp is:")
        # print(temp.shape)
        q_eval = temp.gather(1, actions).squeeze(1)
        #q_eval = self.eval_net(obs).gather(1, actions).squeeze(1) # so got batch size * score
        # argmax a Q'pi (st+1, a) ,detach()使得梯度不回传,max(1)[0],get the max score of action
        q_next = self.target_net(next_obs).max(1)[0].detach() # batch size * score
        # y = rt + γ * (1 - done) * max(Q(s', a')) , if done then no st+1
        q_target = rewards + self.gamma * (1 - dones) * q_next
        # double dqn can be added 

        loss = self.loss_fn(q_eval, q_target) # batch size * 1
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def main():
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)

    # epsilon can decay
    for i_episode in range(args.n_episodes): # 一个episode一把
        obs = env.reset()
        # print(obs.shape)
        episode_reward = 0
        done = False
        step_cnt = 0

        while not done and step_cnt < 500: # 一把到500分或者输就结束
            step_cnt += 1
            env.render()
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done) # 收集学习资料
            episode_reward += reward
            obs = next_obs

            if agent.buffer.len() >= args.batch_size: # 128
                agent.learn() # 收集够资料就学习

        print(f"Episode: {i_episode}, Reward: {episode_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="environment name")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--hidden", default=64, type=int, help="dimension of hidden layer")
    parser.add_argument("--n_episodes", default=500, type=int, help="number of episodes")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--capacity", default=10000, type=int, help="capacity of replay buffer")
    parser.add_argument("--eps", default=0.1, type=float, help="epsilon of ε-greedy")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--update_target", default=100, type=int, help="frequency to update target network")
    args = parser.parse_args()
    main()