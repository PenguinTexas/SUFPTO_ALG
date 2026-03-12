import math
c = 3e8  # 光速m/s
fc = 2e9  # 载波频率(Hz),2GHz
ksi_Los = 3  # ξ_Los损耗衰减因子(dB)
ksi_NLos = 23  # ξ_NLos损耗衰减因子(dB)
P_tr_j = 0.1  # 用户的发射功率(w)
P_tr_k = 1  # 无人机的发射功率(w)
sigma_square = 1e-14  # (w) 噪声功率(-110dBm),此处再议
f_unit = 1000  # 1位任务所需的CPU周期(cycle/bit)


# 无人机与组内用户之间的直线距离d(m)
def distance(j, Group):
    return ((Group.chloc[0] - Group.users[j].Wj[0]) ** 2 + (
            Group.chloc[1] - Group.users[j].Wj[1]) ** 2 + Group.drone_k.hk ** 2) ** 0.5


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


# 上行链路速率R_up:用户将任务上传给无人机的速率
def R_up(j, Group):
    return Group.drone_k.Bk / len(Group.users) * math.log(1 + P_tr_j / (average_path_loss(j, Group) * sigma_square), 2)


# 下行链路速率R_down:无人机将任务结果返回给用户的速率
def R_down(j, Group):
    return Group.drone_k.Bk / len(Group.users) * math.log(1 + P_tr_k / (average_path_loss(j, Group) * sigma_square), 2)


# # 任务卸载以及计算和通信时延属性
# def calculate_time(i, Group):


# 本地计算时延T_loc(cycle/MHz):完成(1-αj)Dj位任务的总延迟
def T_loc(j, Group):
    return (f_unit * (1 - Group.users[j].alpha) * Group.users[j].Dj) / Group.users[j].f_j


# 任务卸载时延T_up:用户j将任务发送给无人机k并进行调度和卸载任务处理所需的时延
def T_up(j, Group):
    return Group.users[j].alpha * Group.users[j].Dj / R_up(j, Group)


# 任务结果返回时延T_down:无人机一旦获得任务的结果，立即将结果返回给相应的用户
def T_down(j, Group):
    return Group.users[j].alpha * Group.users[j].Vj / R_down(j, Group)


# 无人机服务器中的任务计算时延T_comp:无人机k在组i内为用户j的任务计算所需的时延
def T_comp(j, Group):
    return f_unit * Group.users[j].alpha * Group.users[j].Dj / Group.drone_k.f_k


# 无人机服务总时长T_ser:上传任务+计算任务+返回结果
def T_ser(j, Group):
    return T_up(j, Group) + T_comp(j, Group) + T_down(j, Group)


# 飞行时间
def Eta(Q0, Q1, group):  # 上一个组的中心点，此组的中心点，此组
    return (((Q1[0] - Q0[0]) ** 2 + (Q1[1] - Q0[1]) ** 2) ** 0.5) / group.drone_k.Vk


# 无人机服务于用户j之前的用户的等待时延T_wait:
# 无人机k为Ck中优先级高于Cik的组服务所花费的时延(包括飞行时间)，以及无人机k服务于Cik内优先级高于用户j的其他用户所花费的时延。

def T_wait(j, Group):
    # group_list = [Group.drone_k.serve_group[j].h for j in range(len(Group.drone_k.serve_group))]
    waiting_delay = 0
    Q0 = (0, 0)  # 无人机的初始位置
    for group in Group.drone_k.serve_group:
        if group != Group:
            for k in range(len(group.users)):  # 无人机k为Ck中优先级高于Cik的组服务所花费的时延(包括飞行时间)
                waiting_delay += T_ser(k, group)
            # # 加上飞行时间
            waiting_delay += Eta(Q0, group.chloc, group)
            Q0 = group.chloc
        else:
            for k in range(j):  # 无人机k服务于Cik内优先级高于用户j的其他用户所花费的时延。
                waiting_delay += T_ser(k, group)
            # 飞向这一组的时间
            waiting_delay += Eta(Q0, group.chloc, group)
            # Q0 = group.chloc
            return waiting_delay


# 总服务时间T_total:由于在本地计算模式和边缘计算模式下执行的任务Mj的两部分是并行处理的，因此用户j在Ci k内的任务处理总延迟表示为max(T_loc,T_edge)
def T_total(j, Group):
    # a = T_loc(j, Group)
    # b = T_ser(j, Group)
    # c = T_wait(j, Group)
    if Group.users[j].alpha == 0:  # 本地计算
        return T_loc(j, Group)
    elif Group.drone_k is None:  # 边缘计算，但没有分配到无人机,完不成任务，所以为无穷大时间
        return float('inf')
    else:  # 边缘计算
        return max(T_loc(j, Group), T_ser(j, Group) + T_wait(j, Group))


# 能耗属性

# 上传能耗E_up:用户上传任务的能量消耗
def E_up(j, Group):
    return P_tr_j * T_up(j, Group)


# 下行能耗E_down:无人机k将任务结果返回给用户的能耗E_down
def E_down(j, Group):
    return P_tr_k * T_down(j, Group)


# 计算能耗E_comp:
def E_comp(j, Group):
    return Group.drone_k.P_comp_k * T_comp(j, Group)


# 相关参数设置
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


# 无人机k的飞行功率表示:
def P_f_k(Vk):
    P_f = 0.5 * d0 * rho * Rs * A * Vk ** 3 + P0 * (1 + (3 * Vk ** 2) / (omega_e * Re) ** 2) + P1 * (
            (1 + Vk ** 4 / (4 * v0 ** 4)) ** 0.5 - Vk ** 2 / (2 * v0 ** 2)) ** 0.5
    return P_f


# 飞行能耗:E_flight:从上一个组飞到当前组的消耗
def E_flight(Group):
    Q0 = (0, 0)  # 初始化无人机的在上一个组位置
    for group in Group.drone_k.serve_group:
        if group != Group:  # 找到上个组的点
            Q0 = group.chloc
        else:
            return P_f_k(group.drone_k.Vk) * Eta(Q0, group.chloc, group)


# 悬停能耗:E_hover
def E_hover(j, Group):
    return P_f_k(0) * T_ser(j, Group)


# 用户节能属性:假设上传给无人机计算的这部分任务通过用户本地所计算消耗的能耗
def E_scomp(j, Group):
    return Group.users[j].P_comp_j * f_unit * Group.users[j].alpha * Group.users[j].Dj / Group.users[j].f_j


# 节约的能量=假设上传给无人机计算的这部分任务通过用户本地所计算消耗的能耗-上传这部分任务的能耗。
def E_save(j, Group):
    # if Group.users[j].alpha == 0:
    #     return 0
    return E_scomp(j, Group) - E_up(j, Group)


# 最大节能的时候是αj=1:所有任务全部上传到无人机
# 最大节能E_msave:
def E_msave(j, Group):
    return Group.users[j].P_comp_j * f_unit * 1 * Group.users[j].Dj / Group.users[j].f_j - P_tr_j * 1 * Group.users[
        j].Dj / R_up(j, Group)


# 从当前组返回终点(起点)的能耗E_back:
def E_back(Group):
    return P_f_k(Group.drone_k.Vk) * Eta(Group.chloc, (0, 0), Group)


# 为无人机k服务的第i个组内的所有用户的总能耗E_i_k:
def E_i_k(Group):
    E_down_total = E_comp_total = E_flight_total = E_hover_total = 0
    for j in range(len(Group.users)):
        E_down_total += E_down(j, Group)
        E_comp_total += E_comp(j, Group)
        E_hover_total += E_hover(j, Group)
    E_flight_total = E_flight(Group)

    return E_down_total + E_comp_total + E_flight_total + E_hover_total


# 用户满意度属性
lambda_value = 2  # 满意度参数
gamma = 0.5  # 满意度参数


# 计算满意度
def calculate_user_satisfaction(j, Group):
    alpha = Group.users[j].alpha
    Erj = Group.users[j].Erj
    Emaxj = Group.users[j].Emaxj
    beta_j = Group.users[j].beta_j
    Tj = Group.users[j].Tj
    T_total_value = T_total(j, Group)
    T_diff = T_total_value - Tj
    if alpha != 0 and Group.drone_k is None:  # 没有分配组
        return 0
    if Erj >= gamma * Emaxj:
        if T_total_value <= beta_j * Tj:
            if T_diff <= 0:
                return lambda_value - 1
            else:
                return lambda_value - lambda_value ** (T_diff / (beta_j * Tj))
        else:
            return 0
    elif 0 < Erj < gamma * Emaxj:
        if alpha == 0:  # 本地计算，E_save=0所以满意度为零
            return 0
        E_msave_diff = E_msave(j, Group) - E_save(j, Group)
        E_msave_value = E_msave(j, Group)
        if T_total_value <= beta_j * Tj:
            if T_diff <= 0:
                return (lambda_value - 1) * (lambda_value - lambda_value ** (E_msave_diff / E_msave_value))
            else:
                return (lambda_value - lambda_value ** (T_diff / (beta_j * Tj))) * (
                        lambda_value - lambda_value ** (E_msave_diff / E_msave_value))
        else:
            return 0
    else:
        return 0


def total_satisfaction(Groups):
    Total_satisfaction = 0
    for g in Groups:
        for j in range(len(g.users)):
            Total_satisfaction += calculate_user_satisfaction(j, g)
    Total_satisfaction = round(Total_satisfaction, 2)
    return Total_satisfaction

