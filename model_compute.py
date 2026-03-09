import math
import numpy as np
import random

# -------------------------相关参数设置------------------------- #
c = 3e8  # 光速m/s
fc = 2e9  # 载波频率(Hz),2GHz
ksi_Los = 3  # ξ_Los损耗衰减因子(dB)
ksi_NLos = 23  # ξ_NLos损耗衰减因子(dB)
P_tr_j = 0.1  # 用户的发射功率(w)
P_tr_k = 1  # 无人机的发射功率(w)
sigma_square = 1e-14  # (w) 噪声功率(-110dBm)
f_unit = 1000  # 1位任务所需的CPU周期(cycle/bit)

delta_e = 0.012  # 附录delta[32]
rho = 1.225  # rho
Rs = 0.05  # s
A = 0.503  # A
omega_e = 300  # omege
Re = 0.4  # R
Tau = 0.1  # k
W = 20  # 重力 g*M_uav
SF = 0.0151

P0 = delta_e / 8 * rho * Rs * A * (omega_e * Re) ** 3
P1 = (1 + Tau) * (W ** (3 / 2) / (2 * rho * A) ** 0.5)  # 公式(64)[32]
v0 = (W / (2 * rho * A)) ** 0.5
d0 = SF / (Rs * A)

# -------------------------通信模型------------------------- #
# 视线路径损失链路l_Los_i_j_k(dB):在无人机与用户之间的视线范围内的信道损失链路
def l_los(j, Group):
    return (4 * math.pi * fc * distance(j, Group) / c) ** 2 * ksi_Los

# 非视线路径损失链路l_NLos_i_j_k(dB):在无人机与用户之间的非视线范围内的信道损失链路
def l_Nlos(j, Group):
    return (4 * math.pi * fc * distance(j, Group) / c) ** 2 * ksi_NLos

# LOS链路概率P_Los:计算视线路径链路的概率
def P_Los(j, Group):
    Theta = math.asin(Group.drone_k.hk / distance(j, Group))  # 仰角
    a, b = 11.95, 0.14  # 环境参数
    return 1 / (1 + a * math.exp(-b * (180 / math.pi * Theta - a)))

# NLos链路概率P_NLos:计算非视线路径链路的概率
def P_NLos(j, Group):
    return 1 - P_Los(j, Group)

# 平均路径损耗l_i,j,k(dB):计算信号传输的平均损耗
def average_path_loss(j, Group):
    return l_los(j, Group) * P_Los(j, Group) + l_Nlos(j, Group) * P_NLos(j, Group)

# 带宽大小：上行链路速率R_up:用户将任务上传给无人机的速率
def R_up(j, Group):
    # return Group.drone_k.Bk / len(Group.users) * math.log(1 + P_tr_j / (average_path_loss(j, Group) * sigma_square), 2)
    return Group.drone_k.Bk * math.log(1 + P_tr_j / (average_path_loss(j, Group) * sigma_square), 2)

# 带宽大小：下行链路速率R_down:无人机将任务结果返回给用户的速率
def R_down(j, Group):
    # return Group.drone_k.Bk / len(Group.users) * math.log(1 + P_tr_k / (average_path_loss(j, Group) * sigma_square), 2)
    return Group.drone_k.Bk * math.log(1 + P_tr_k / (average_path_loss(j, Group) * sigma_square), 2)

# -------------------------计算模型------------------------- #
# 本地计算时延T_loc(cycle/MHz):完成(1-αj)Dj位任务的总延迟
def T_loc(j, Group):
    return (f_unit * (1 - Group.users[j].alpha) * Group.users[j].Dj) / Group.users[j].f_j

# 无人机服务器中的任务计算时延T_comp:无人机k在组i内为用户j的任务计算所需的时延
def T_comp(j, Group):
    return f_unit * Group.users[j].alpha * Group.users[j].Dj / Group.drone_k.f_k

# 任务卸载时延T_up:用户j将任务发送给无人机k并进行调度和卸载任务处理所需的时延
def T_up(j, Group):
    return Group.users[j].alpha * Group.users[j].Dj / R_up(j, Group)

# 任务结果返回时延T_down:无人机一旦获得任务的结果，立即将结果返回给相应的用户
def T_down(j, Group):
    return Group.users[j].alpha * Group.users[j].Vj / R_down(j, Group)

# 当前组的悬停的总时间，就是所有用户接受服务的时间，在这个时间内进行双机流水作业
# 返回第j个用户任务完成时的时间
def T_ser(j, Group):
    # 创建一个数组来存储每个任务的上传和计算时间
    task_uptime_ctime = [[0, 0] for _ in range(len(Group.users))]
    for i in range(len(Group.users)):
        task_uptime_ctime[i][0] = T_up(i, Group)
        task_uptime_ctime[i][1] = T_comp(i, Group) + T_down(i, Group)

    compute_array = []
    # 计算双机时间
    NIC_time = task_uptime_ctime[0][0]
    CPU_time = task_uptime_ctime[0][0] + task_uptime_ctime[0][1]
    compute_array.append(CPU_time)
    for i in range(1, len(Group.users)):
        NIC_time += task_uptime_ctime[i][0]
        CPU_time = CPU_time + task_uptime_ctime[i][1] if NIC_time < CPU_time else NIC_time + task_uptime_ctime[i][1]
        compute_array.append(CPU_time)
    # print(compute_array)
    return compute_array[j]

# 飞行时间
def Eta(Q0, Q1, group):  # 上一个组的中心点，此组的中心点，此组
    return (((Q1[0] - Q0[0]) ** 2 + (Q1[1] - Q0[1]) ** 2) ** 0.5) / group.drone_k.Vk

# 组Group的服务开始时间
def T_Group_Start(Group):
    waiting_delay = 0
    Q0 = (0, 0)  # 无人机的初始位置
    for group in Group.drone_k.serve_group:
        if group != Group:
            # 无人机k为当前的组的前面的组服务时所花费的时间，包括飞行时间
            # 就是加上该组的最后一个用户服务完成时间
            waiting_delay += Eta(Q0, group.chloc, group)
            Q0 = group.chloc
            waiting_delay += T_ser(len(group.users) - 1, group)
        else:
            # 飞向这一组的时间
            waiting_delay += Eta(Q0, group.chloc, group)
            # Q0 = group.chloc
            break  # 找到目标组，退出循环
    return waiting_delay

# 总服务时间T_total:由于在本地计算模式和边缘计算模式下执行的任务Mj的两部分是并行处理的，因此用户j在Ci k内的任务处理总延迟表示为max(T_loc,T_edge)
def T_total(j, Group):
    if Group.users[j].alpha == 0:  # 本地计算
        return T_loc(j, Group)
    elif Group.drone_k is None:  # 边缘计算，但没有分配到无人机,完不成任务，所以为无穷大时间
        return float('inf')
    else:  # 边缘计算
        return max(T_loc(j, Group), T_Group_Start(Group) + T_ser(j, Group))

# -------------------------能耗模型------------------------- #
# 上传能耗E_up:用户上传任务的能量消耗
def E_up(j, Group):
    return P_tr_j * T_up(j, Group)

# 下行能耗E_down:无人机k将任务结果返回给用户的能耗E_down
def E_down(j, Group):
    return P_tr_k * T_down(j, Group)

# 计算能耗E_comp:
def E_comp(j, Group):
    return Group.drone_k.P_comp_k * T_comp(j, Group)

# 无人机k的飞行功率表示:
def P_f_k(Vk):
    P_f = (0.5 * d0 * rho * Rs * A * Vk ** 3 + P0 * (1 + (3 * Vk ** 2) / (omega_e * Re) ** 2) +
           P1 * ((1 + Vk ** 4 / (4 * v0 ** 4)) ** 0.5 - Vk ** 2 / (2 * v0 ** 2)) ** 0.5)
    return P_f

# 飞行能耗:E_flight:从上一个组飞到当前组的消耗
def E_flight(Group):
    Q0 = (0, 0)  # 初始化无人机的在上一个组位置
    for group in Group.drone_k.serve_group:
        if group != Group:  # 找到上个组的点
            Q0 = group.chloc
        else:
            break  # 找到目标组，退出循环
    return P_f_k(Group.drone_k.Vk) * Eta(Q0, Group.chloc, Group)

# 悬停能耗:E_hover
def E_hover(Group):
    return P_f_k(0) * T_ser(len(Group.users) - 1, Group)

# 用户节能属性:假设上传给无人机计算的这部分任务通过用户本地所计算消耗的能耗
def E_scomp(j, Group):
    return Group.users[j].P_comp_j * f_unit * Group.users[j].alpha * Group.users[j].Dj / Group.users[j].f_j

# 节约的能量=假设上传给无人机计算的这部分任务通过用户本地所计算消耗的能耗-上传这部分任务的能耗。
def E_save(j, Group):
    return E_scomp(j, Group) - E_up(j, Group)

# 最大节能的时候是αj=1:所有任务全部上传到无人机
def E_msave(j, Group):
    return (Group.users[j].P_comp_j * f_unit * 1 * Group.users[j].Dj / Group.users[j].f_j -
            P_tr_j * 1 * Group.users[j].Dj / R_up(j, Group))

# 从当前组返回初始点的能耗E_back:
def E_back(Group):
    return P_f_k(Group.drone_k.Vk) * Eta(Group.chloc, (0, 0), Group)

# 无人机视角：无人机k服务的第i个组内的所有用户的总能耗E_i_k:
def E_i_k(Group):
    E_down_total = E_comp_total = E_flight_total = E_hover_total = 0
    for j in range(len(Group.users)):
        E_down_total += E_down(j, Group)
        E_comp_total += E_comp(j, Group)
    E_hover_total = E_hover(Group)
    E_flight_total = E_flight(Group)
    return E_down_total + E_comp_total + E_flight_total + E_hover_total

# -------------------------用户满意度模型------------------------- #
lambda_value_1 = 2  # 满意度参数
lambda_value_2 = 2  # 满意度参数
gamma = 0.8  # 满意度参数

# S_1：与任务延迟相关的用户满意度
def S_1(j, Group):
    beta_j = Group.users[j].beta_j
    Tj = Group.users[j].Tj     # 用户期望的最佳任务完成时间
    T_total_value = T_total(j, Group)   # 用户的实际任务完成时间
    T_diff = T_total_value - Tj     # 差值

    # S_1函数
    if T_total_value <= beta_j * Tj:
        if T_diff <= 0:
            S_a = 1
        else:
            S_a = (lambda_value_1 - lambda_value_1 ** (T_diff / (beta_j * Tj - Tj))) / (lambda_value_1 - 1)
    else:
        S_a = 0
    return S_a

# S_2：与节能效果相关的用户满意度
def S_2(j, Group):
    alpha = Group.users[j].alpha
    Erj = Group.users[j].Erj       # 剩余能量
    Emaxj = Group.users[j].Emaxj    # 总能量

    if Erj >= gamma * Emaxj:
        S_e = 1
    elif 0 < Erj < gamma * Emaxj:
        if alpha == 0:  # 本地计算，E_save=0所以满意度为零
            S_e = 0
        else:
            S_e = (lambda_value_2 - lambda_value_2 ** (1 - (E_save(j, Group) / E_msave(j, Group)))) / (lambda_value_2 - 1)
    else:
        S_e = 0
    return S_e

# 用户满意度通过综合考虑任务延迟和节能效果来评估
def calculate_user_satisfaction(j, Group):
    if Group.users[j].alpha != 0 and Group.drone_k is None:  # 卸载决策不为0且没有分配组，完不成任务
        return 0
    else:
        return S_1(j, Group) * S_2(j, Group)

# 指定组的用户的总满意度
def total_satisfaction(Groups):
    sum_satisfaction = 0
    for g in Groups:
        for j in range(len(g.users)):
            sum_satisfaction += calculate_user_satisfaction(j, g)
    sum_satisfaction = round(sum_satisfaction, 2)
    return sum_satisfaction

# 指定组的所有用户的平均满意度
def aver_satisfaction(Groups, users):
    return total_satisfaction(Groups) / len(users)

# 指定无人机所服务组的总满意度
def calculate_satisfaction_by_drones(drones):
    total = 0
    for d in drones:
        for g in d.serve_group:
            for j in range(len(g.users)):
                total += calculate_user_satisfaction(j, g)
    return total

# -------------------------通用函数------------------------- #
# 计算两点之间的欧氏距离
def calculate_distance(point1, point2):
    if len(point1) == 3 and len(point2) == 3:
        # 3D坐标距离计算
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)
    else:
        # 2D坐标距离计算
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# 用户与无人机的距离
def distance(j, Group):
    return calculate_distance(
        (Group.chloc[0], Group.chloc[1], Group.drone_k.hk),
        (Group.users[j].Wj[0], Group.users[j].Wj[1], 0)
    )

# 无人机提供服务的能量约束
def energy_constraint(drone):
    E_total = 0  # 消耗加上返回的能量
    for g in drone.serve_group:
        E_total += E_i_k(g)
    E_total += E_back(drone.serve_group[-1])
    if E_total >= drone.Emaxk:  # 如果能量不够则返回false
        # print(f"消耗大于无人机最大能量,无人机总能耗:{E_total},无人机最大能量:{drone.Emaxk}")
        return False
    else:
        # print(f"消耗小于无人机最大能量,无人机总能耗:{E_total},无人机最大能量:{drone.Emaxk}")
        return True

# 检查各无人机总能量约束
def energy_total_constraint(drones):
    for drone in drones:
        if not energy_constraint(drone):
            return False
    return True

# 修正卸载比例满足能量约束
def offloading_modify_method(group, solution):
    solution = solution.copy()  # 避免修改原始解
    for _ in range(100):  # 最大尝试次数
        # 找出所有最大alpha值的索引
        max_alpha = max(solution)
        max_indices = [i for i, v in enumerate(solution) if v == max_alpha]
        # 随机选择一个最大alpha进行减小
        idx = random.choice(max_indices)
        reduction = random.uniform(min(solution), max_alpha)
        solution[idx] = max(solution[idx] - reduction, 0)  # 确保不小于0

        # 更新用户alpha值并检查约束
        for u, alpha in zip(group.users, solution):
            u.alpha = alpha
        if energy_constraint(group.drone_k):
            return solution

    # 100次尝试后仍未满足约束
    zero_solution = np.zeros(len(group.users))
    for u, alpha in zip(group.users, zero_solution):
        u.alpha = alpha
    return zero_solution if energy_constraint(group.drone_k) else False