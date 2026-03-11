import os
import numpy as np
from gym import spaces
import torch
import torch.nn.functional as F
import rl_utils
from PSO_based_UAV_Scheduling_Algorithm import PSO_UAV_Scheduling
from PSO_based_UAV_Offloading_Scheduling import offloading_modify_method
from model_compute import calculate_user_satisfaction, E_i_k, E_back, energy_total_constraint, total_satisfaction


# 无人机超出能量则施加惩罚:奖励函数
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
            # 可以根据超出的程度和具体情况设定惩罚值
            total_beyond.append(E_total - E_max)
    return total_beyond

# 根据用户组的信息计算总用户满意度
def cal_user_satisfaction(groups):
    sum_satisfaction = 0
    for g in groups:
        for j in range(len(g.users)):
            sum_satisfaction += calculate_user_satisfaction(j, g)
    return round(sum_satisfaction, 1)

# -------------------------环境设置------------------------- #
class TaskAllocationEnv:
    def __init__(self, drones, users, groups):
        self.drones = drones      # 服务的无人机对象
        self.users = users
        self.groups = groups    # 所有待服务的用户组列表
        self.num_users = len(users)
        self.num_drones = len(drones)
        self.num_groups = len(groups)
        # 定义动作空间 (动作空间为所有用户的卸载比例值)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_users,))
        # 定义状态空间 (动态拓展)
        self.state = self._get_state()
        self.observation_space = spaces.Box(low=0, high=1, shape=self.state.shape)
        self.max_steps = 100  # 设定的最大时间步数
        self.reset()  # 重置环境
    # 生成状态向量
    def _get_state(self):
        # 用户信息
        user_info = np.array(
            [[user.Dj, user.Vj, user.Tj, user.T, user.P_comp_j, user.f_j, user.beta_j, user.xj, user.yj]
             for user in
             self.users])

        # 用户组信息
        Group_satisfaction = []
        for group in self.groups:
            total_satisfaction = 0
            for j, user in enumerate(group.users):
                total_satisfaction += calculate_user_satisfaction(j, group)
            Group_satisfaction.append(total_satisfaction)
        group_info = np.array(Group_satisfaction)

        # 无人机信息
        E_drones = []
        for drone in self.drones:
            E_total = 0  # 消耗加上返回的能量
            for g in drone.serve_group:
                E_total += E_i_k(g)
            E_total += E_back(drone.serve_group[-1])
            E_drones.append(E_total)

        # 归一化
        user_info = (user_info - user_info.min(axis=0)) / (user_info.max(axis=0) - user_info.min(axis=0))
        group_info = (group_info - min(group_info)) / (max(group_info) - min(group_info))

        return np.concatenate((user_info.flatten(), group_info.flatten()))
    # 更新
    def step(self, action):
        # 执行动作，更新环境状态
        for i, user in enumerate(self.users):
            user.alpha = action[i]  # 更新用户的任务卸载比例
        # 判断是否违反能量约束
        total_beyond_list = energy_constraint_penalty(self.drones)
        if len(total_beyond_list) > 0:
            penalty = len(total_beyond_list) * -(cal_user_satisfaction(self.groups)) * 0.1
        else:
            penalty = 0
        # 计算奖励，这里使用所有用户满意度之和作为奖励值
        reward = (cal_user_satisfaction(self.groups) + penalty) / 10
        # 判断是否结束
        self.current_step += 1  # 增加时间步数
        if self.current_step >= self.max_steps:
            done = True  # 到达最大步数，设置为终止状态
        else:
            done = False
        # 返回下一个状态、奖励、是否结束和额外信息
        next_state = self._get_state()
        return next_state, reward, done, {}
    # 初始化重置用户的任务卸载比例等信息
    def reset(self):
        for u in self.users:
            u.alpha = 1
        self.current_step = 0
        self.state = self._get_state()
        return self.state
# -------------------------环境设置------------------------- #
#  构造策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        # self.fc1 是一个 全连接层，将输入 x 从 state_dim 维映射到 hidden_dim 维
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)  # 添加额外的隐藏层
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):    # 缩放至动作范围
        x = F.relu(self.fc1(x))     # ReLU 的特性：保留正输入，负输入置零
        x = F.relu(self.fc2(x))
        x = (torch.tanh(self.fc3(x)) + 1) / 2
        return x
# 状态-动作价值网络，输出状态动作对 (状态，动作) 的价值
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作，需要将两者同时输入网络
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_out(x)  # 返回状态-动作对的预测价值
# 主算法
class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 直接更新 Actor/Critic 会导致训练不稳定，目标网络通过缓慢更新（软更新）提供稳定的目标值
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 将目标网络的参数初始化为与在线网络（actor/critic）相同，确保训练初期的一致性
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        # 优化器分别更新 Actor 和 Critic 网络，通常 actor_lr < critic_lr（Critic 需要更快收敛以提供准确的 Q-value）
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  # 影响智能体对长期奖励的重视程度（接近 1 表示更关注未来）
        self.sigma = sigma  # 控制探索噪声的大小（高斯噪声的标准差）
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
        action = np.clip(action, 0, 1)
        return action

    def soft_update(self, net, target_net):
        # 目标网络软更新遍历在线网络（net）和目标网络（target_net）的所有参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 超参数tau，较小的tau训练更稳定，但收敛慢
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        # transition_dict 是从经验回放池（Replay Buffer）采样的一批数据，并从 NumPy 数组转换为 PyTorch Tensor
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        '''Critic 网络更新:目标是 最小化 TD 误差'''
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))  # 计算当前 Critic 预测的 Q 值和目标 Q 值的均方误差（MSE）
        # 反向传播-梯度下降，学习率为超参数
        self.critic_optimizer.zero_grad() # 清空梯度
        critic_loss.backward() #  计算梯度
        self.critic_optimizer.step() # 更新 Critic 网络参数
        '''Actor 网络更新:找到最优策略,最大化 Critic 的 Q 值'''
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))  # 取负均值（因为优化器默认是 最小化 损失，而我们需要 最大化 Q 值）
        # 反向传播-梯度下降
        self.actor_optimizer.zero_grad() # 清空梯度
        actor_loss.backward() #  计算梯度
        self.actor_optimizer.step() # 更新 Actor 网络参数

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
# -------------------------训练测试------------------------- #
# 训练
def DDPG_train(users, drones, Groups):
    # todo:参数设置, 可根据模型效果微调
    actor_lr = 1e-4
    critic_lr = 1e-3
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 500
    batch_size = 256
    sigma = 0.01  # 高斯噪声标准差
    # 无人机 - 用户组预匹配
    drones, Groups, schedule_fitness = PSO_UAV_Scheduling(drones, Groups)
    drone_number = len(drones)
    users_number = len(users)
    # 训练准备, 创建 环境 和 Agent 实例
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = TaskAllocationEnv(drones, users, Groups)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    # 训练策略
    return_list, max_satisfaction, action = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, drone_number, users_number)
    os.makedirs(f'model/{len(drones)}_{len(users)}', exist_ok=True)
    # 保存 Actor 和 Critic 的模型参数
    torch.save(agent.actor.state_dict(), f'model/{len(drones)}_{len(users)}/actor_model.pth')
    torch.save(agent.critic.state_dict(), f'model/{len(drones)}_{len(users)}/critic_model.pth')
    print(f"最终最大的奖励：{max_satisfaction * 10}")
    print(f"动作：{action}")
    return max_satisfaction * 10, Groups, drones
# 测试
def DDPG_test(users, drones, Groups):
    # 无人机 - 用户组预匹配
    drones, Groups, schedule_fitness = PSO_UAV_Scheduling(drones, Groups)
    drone_number = len(drones)
    users_number = len(users)
    # 创建 环境 和 Agent 实例
    env = TaskAllocationEnv(drones, users, Groups)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    hidden_dim = 128
    # 创建 Actor 模型
    actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
    # 加载保存的模型参数到 Actor 模型中
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
    # 检查能量约束，如不满足则按组调整
    if not energy_total_constraint(drones):
        print("能量约束不满足，调整卸载比例...")
        # 创建用户到 action 索引的映射
        user_to_idx = {id(u): i for i, u in enumerate(users)}

        # 按组调整卸载比例
        for g in Groups:
            if g.drone_k is not None and len(g.users) > 0:
                # 提取该组用户对应的 action 子集
                group_action = np.array([action[user_to_idx[id(u)]] for u in g.users])
                # 调用修正函数
                offloading_modify_method(g, group_action)

        print("经过卸载比例调整后，测试完成")
    else:
        print("测试完成")
    satisfaction = total_satisfaction(Groups)
    return satisfaction, Groups, drones
