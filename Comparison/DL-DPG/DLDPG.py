import collections
import pickle
import random
import sys
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from GA_based_Task_Offloading_Algorithm import energy_total_constraint
import gym
from gym import spaces
import numpy as np
from GA_based_UAV_Scheduling_Algorithm import GA_based_UAV_Scheduling_Algorithm
from model_compute import calculate_user_satisfaction, E_i_k, E_back, total_satisfaction
import os

matplotlib.use('TkAgg')  # 设置使用 TkAgg 后端，也可以尝试其他后端
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def energy_constraint(drone):
    E_total = 0  # 消耗加上返回的能量
    for g in drone.serve_group:
        E_total += E_i_k(g)
    E_total += E_back(drone.serve_group[-1])
    if E_total >= drone.Emaxk:  # 如果能量不够则返回false
        return False
    else:
        return True


def offloading_modify_method(users, solution, drones):  # 修正成满足能量约束
    attempts = 0
    while True:
        max_alpha = max(solution)
        all_max_index = [i for i, v in enumerate(solution) if v == max_alpha]
        max_index = random.choice(all_max_index)
        min_alpha = min(solution)
        solution[max_index] = max(solution[max_index] - random.uniform(min_alpha, max_alpha), 0)
        for u, s in zip(users, solution):
            u.alpha = s
        if energy_total_constraint(drones):
            return solution
        else:
            attempts += 1
            if attempts == 100:
                print(solution)
                for u, s in zip(users, np.zeros(len(users), dtype=float)):
                    u.alpha = s
                if energy_total_constraint(drones):
                    return np.zeros(len(users), dtype=float)
                else:
                    return False


def energy_constraint_penalty(drones):
    total_beyond = []
    for drone in drones:
        E_total = 0
        for g in drone.serve_group:
            E_total += E_i_k(g)
        E_total += E_back(drone.serve_group[-1])
        # 假设能量约束为 E_max，如果能量超出范围则施加惩罚
        E_max = drone.Emaxk
        if E_total > E_max:
            print(f"total:{E_total},E_max:{E_max}")
            total_beyond.append(E_total - E_max)

    return total_beyond


class TaskAllocationEnv:
    def __init__(self, drones, users, groups):
        self.drones = drones
        self.users = users
        self.groups = groups
        self.num_users = len(users)
        self.num_drones = len(drones)
        self.num_groups = len(groups)
        self.reset()  # 重置环境
        # 定义动作空间（假设每个用户的任务卸载比例在[0, 1]之间）
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_users,))
        # 定义状态空间
        self.state = self._get_state()
        self.observation_space = spaces.Box(low=0, high=1, shape=self.state.shape)
        self.max_steps = 100  # 设定的最大时间步数

    def _reset_env(self):
        # 初始化用户的任务卸载比例等信息
        for u in self.users:
            u.alpha = 1

        self.current_step = 0  # 当前时间步数
        self.state = self._get_state()

    def _get_state(self):
        user_info = np.array(
            [[user.Dj, user.Vj, user.Tj, user.T, user.P_comp_j, user.f_j, user.beta_j, user.xj, user.yj]
             for user in
             self.users])

        Group_s = []
        for g in self.groups:
            total = 0
            for j, u in enumerate(g.users):
                total += calculate_user_satisfaction(j, g)
            Group_s.append(total)
        group_info = np.array(Group_s)

        E_drones = []
        for drone in self.drones:
            E_total = 0  # 消耗加上返回的能量
            for g in drone.serve_group:
                E_total += E_i_k(g)
            E_total += E_back(drone.serve_group[-1])
            E_drones.append(E_total)

        user_info = (user_info - user_info.min(axis=0)) / (user_info.max(axis=0) - user_info.min(axis=0))
        group_info = (group_info - min(group_info)) / (max(group_info) - min(group_info))
        return np.concatenate((user_info.flatten(), group_info.flatten()))

    def cal_user_satisfaction(self):
        sum_satisfaction = 0
        # 根据用户组的信息计算总用户满意度
        for g in self.groups:
            for j in range(len(g.users)):
                sum_satisfaction += calculate_user_satisfaction(j, g)

        return round(sum_satisfaction, 1)

    def step(self, action):
        # 执行动作，更新环境状态
        for i, user in enumerate(self.users):
            user.alpha = action[i]  # 更新用户的任务卸载比例
        # 判断是否违反能量约束
        total_beyond_list = energy_constraint_penalty(self.drones)
        if len(total_beyond_list) > 0:
            penalty = len(total_beyond_list) * -(self.cal_user_satisfaction()) * 0.1
        else:
            penalty = 0
        # 计算奖励，这里使用所有用户满意度之和作为奖励值
        reward = (self.cal_user_satisfaction() + penalty) / 10
        # 判断是否结束
        self.current_step += 1  # 增加时间步数
        # 判断是否达到最大时间步数
        if self.current_step >= self.max_steps:
            done = True  # 到达最大步数，设置为终止状态
        else:
            done = False
        # 返回下一个状态、奖励、是否结束和额外信息
        next_state = self._get_state()
        return next_state, reward, done, {}

    def reset(self):
        # 调用环境重置方法
        self._reset_env()
        return self.state


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)  # 添加额外的隐藏层
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = (torch.tanh(self.fc4(x)) + 1) / 2
        return x


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc_out(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class DLDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.time = 0

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        # 给动作添加噪声，增加探索
        random_noise = np.random.uniform(-0.01, 0.01, size=self.action_dim)
        # 生成服从标准正态分布的噪声
        normal_noise = self.sigma * np.random.randn(self.action_dim)
        action += normal_noise + random_noise
        action = np.clip(action, 0, 1)  # 裁剪动作到 [0, 1] 范围内
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, name1, name2):
    return_list = []
    a = 0
    max_reward = float('-inf')
    best_action = None
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                start_time = time.time()
                episode_return = 0
                state = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    # 记录最大奖励对应的动作
                    if reward > max_reward:
                        max_reward = reward
                        best_action = action
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                episodes_list = list(range(len(return_list)))
                plt.plot(episodes_list, return_list)
                plt.xlabel('Episodes')
                plt.ylabel('Returns')
                plt.title('DLDPG')
                plt.savefig(f'dldpg_{name1}_{name2}.jpg', dpi=300)
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % episode_return,
                                  'i_episode_time': '%.2f' % (time.time() - start_time)
                                  })
                pbar.update(1)
    plt.close()
    return return_list, max_reward, best_action


def DLDPG_train(users, drones, Groups):
    drones, Groups = GA_based_UAV_Scheduling_Algorithm(drones, Groups, 'p', 'GA')
    name1 = len(drones)
    name2 = len(users)
    actor_lr = 1e-4
    critic_lr = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 500
    batch_size = 256
    sigma = 0.01  # 高斯噪声标准差0.01
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = TaskAllocationEnv(drones, users, Groups)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    agent = DLDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    # 训练
    return_list, max_satisfaction, action = train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                                                                   minimal_size,
                                                                   batch_size, name1, name2)
    os.makedirs(f'model/{len(drones)}_{len(users)}', exist_ok=True)
    # 保存 Actor 和 Critic 的模型参数
    torch.save(agent.actor.state_dict(), f'model/{len(drones)}_{len(users)}/actor_model.pth')
    torch.save(agent.critic.state_dict(), f'model/{len(drones)}_{len(users)}/critic_model.pth')
    print(f"最终最大的奖励：{max_satisfaction * 10}")
    print(f"动作：{action}")
    return max_satisfaction * 10, Groups, drones


def DLDPG_test(users, drones, Groups):
    drones, Groups = GA_based_UAV_Scheduling_Algorithm(drones, Groups, 'p', 'GA')
    # 创建环境和 Agent 实例
    env = TaskAllocationEnv(drones, users, Groups)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    hidden_dim = 128
    # 创建 Actor 模型
    actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
    # 加载保存的模型参数到 Actor 模型中
    drone_number = len(drones)
    users_number = len(users)
    actor.load_state_dict(torch.load(f'model/{drone_number}_{users_number}/actor_model.pth'))
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = actor.to(device)
    # 将模型设置为 evaluation 模式
    actor.eval()
    # 运行环境，并使用 Actor 模型执行动作选择
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor([state], dtype=torch.float).to(device)
        with torch.no_grad():
            action = actor(state_tensor).detach().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
    for u, a in zip(users, action):
        u.alpha = a
    if energy_total_constraint(drones):
        print("yes")
    else:
        action = offloading_modify_method(users, action, drones)
        for u, a in zip(users, action):
            u.alpha = a
        print("no,yes")
    satisfaction0 = total_satisfaction(Groups)
    return satisfaction0, Groups, drones, satisfaction