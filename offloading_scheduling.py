import math
import os
import pickle
import random
import sys
import time

import numpy as np

from GA_based_UAV_Scheduling_Algorithm import simulated_annealing
from model_compute import calculate_user_satisfaction, total_satisfaction, calculate_satisfaction_by_drones, \
    aver_satisfaction, E_i_k, E_back, S_1, S_2

max_attempts = 100  # 设置循环的最大尝试次数，能量约束处
def energy_constraint(drone):
    E_total = 0  # 消耗加上返回的能量
    for g in drone.serve_group:
        E_total += E_i_k(g)
    E_total += E_back(drone.serve_group[-1])
    if E_total >= drone.Emaxk:  # 如果能量不够则返回false
        return False
    else:
        return True

# 最后优化卸载决策的函数
def PSO_single_optimization(drones, Groups, users):
    total = 0
    for d in drones:
        for g in d.serve_group:
            # 卸载优化使用sa
            current_solution, fitness = standard_pso(g, 'later')
            print(calculate_satisfaction_by_drones(drones))
            total += fitness
    for g in Groups:
        g_satisfaction = sum(calculate_user_satisfaction(j, g) for j in range(len(g.users)))
        if g_satisfaction == 0 and g.drone_k is not None:
            print("有没必要去服务的组")
            g.drone_k.serve_group = [group for group in g.drone_k.serve_group if group != g]
            g.drone_k = None
            for u in g.users:
                u.alpha = 0
    satisfaction = total_satisfaction(Groups)
    print("最后一次输出：", calculate_satisfaction_by_drones(drones))
    return drones, satisfaction

# fitness
def fitness_function(group, solution):
    for u, s in zip(group.users, solution):
        u.alpha = s
    total_cost = 0
    if not energy_constraint(group.drone_k):
        return False
    for j in range(len(group.users)):
        total_cost += calculate_user_satisfaction(j, group)
    total_cost = round(total_cost / len(group.users), 5)
    return total_cost

# 初始化种群
def offloading_modify_method(group, solution):
    attempts = 0
    while True:
        max_alpha = max(solution)
        all_max_index = [i for i, v in enumerate(solution) if v == max_alpha]
        max_index = random.choice(all_max_index)
        min_alpha = min(solution)
        solution[max_index] = max(solution[max_index] - random.uniform(min_alpha, max_alpha), 0)
        for u, s in zip(group.users, solution):
            u.alpha = s
        if energy_constraint(group.drone_k):
            return solution
        else:
            attempts += 1
            if attempts == 100:
                print(solution)
                for u, s in zip(group.users, np.zeros(len(group.users), dtype=float)):
                    u.alpha = s
                if energy_constraint(group.drone_k):
                    return np.zeros(len(group.users), dtype=float)
                else:
                    return False

def initialize_population(group):  # 随机初始卸载率序列
    solution = np.zeros(len(group.users), dtype=float)
    for j, u in enumerate(group.users):
        random_alpha = random.uniform(0, 1)
        u.alpha = random_alpha  # 重新随机一个
        solution[j] = random_alpha
    if not energy_constraint(group.drone_k):  # 判断满足无人机最大能量约束
        solution = offloading_modify_method(group, solution)
    return solution

# 粒子群算法
def standard_pso(Group, run_time):
    num_particles = 100  # 粒子数量
    num_dimensions = len(Group.users)  # 粒子维度
    num_iterations = 100  # 迭代次数
    w_max = 0.9  # 惯性权重的最大值
    w_min = 0.4  # 惯性权重的最小值
    x_max = 1  # 位置最大值
    x_min = 0  # 位置最小值
    v_max = 0.1  # 最大速度
    v_min = -0.1  # 最小速度

    cognitive_weight = 1.5  # 认知权重
    social_weight = 1.5  # 社会权重

    no_change_count = 0  # 记录没有变化的次数
    max_no_change = 10  # 迭代停止不变次数
    best_fitness = 0  # 记录最好的值

    solution_0 = np.array([u.alpha for u in Group.users])
    if fitness_function(Group, solution_0):
        run_time = 'later'  # 原来的还满足就将原来地作为初始解
    else:
        run_time = 'before'  # 否则随机一个满足能量约束的初始解
    if run_time == 'before':

        # 初始化可行解
        solution_list = []
        for _ in range(num_particles):
            solution = initialize_population(Group)
            solution_list.append(solution)

        # 初始化粒子群
        particles = [{'position': s,  # 随机初始化位置
                      'velocity': np.random.uniform(v_min, v_max, num_dimensions),  # 随机初始化速度
                      'best_position': s,  # 个体最佳位置初始化为当前位置
                      'best_fitness': fitness_function(Group, s)}  # 个体最佳适应度
                     for s in solution_list]

    elif run_time == 'later':
        # 初始化粒子群
        particles = [{'position': solution_0.copy(),  # 按照原来初始化位置
                      'velocity': np.random.uniform(v_min, v_max, num_dimensions),  # 随机初始化速度
                      'best_position': solution_0.copy(),  # 个体最佳位置
                      'best_fitness': fitness_function(Group, solution_0)}  # 个体最佳适应度
                     for _ in range(num_particles)]
    else:
        raise Exception("使用了不该使用的run_time指定值")
    # 初始化全局最优解
    global_best = {'position': [], 'fitness': 0}
    if run_time == 'later':
        global_best = {'position': solution_0,
                       'fitness': fitness_function(Group, solution_0)}  # 全局最佳位置和适应度,设成初始值
    else:
        max_fitness = max(p['best_fitness'] for p in particles)
        max_fitness_particle = next(p for p in particles if p['best_fitness'] == max_fitness)
        global_best = {'position': max_fitness_particle['position'], 'fitness': max_fitness}
        print(max_fitness)

    global_best_history = [global_best['position']]  # 记录每一代全局最优解
    for iteration in range(num_iterations):
        start_time = time.time()
        for k, particle in enumerate(particles):
            # 计算适应度
            fitness = fitness_function(Group, particle['position'])
            if fitness is False:
                continue
            fitness = fitness_function(Group, particle['position'])
            # 更新个体最佳位置
            if fitness > particle['best_fitness']:
                particle['best_position'] = particle['position'].copy()
                particle['best_fitness'] = fitness
            # 更新全局最佳位置
            if fitness > global_best['fitness']:
                global_best['position'] = particle['position'].copy()
                global_best['fitness'] = fitness
                global_best_history.append(global_best['position'])
            # 计算动态惯性权重
            inertia_weight = w_max - (w_max - w_min) * (iteration / num_iterations)
            # 惯性部分+认知部分+社会部分
            new_velocity = (inertia_weight * particle['velocity'] +
                            cognitive_weight * np.random.rand(num_dimensions) * (
                                    particle['best_position'] - particle['position']) +
                            social_weight * np.random.rand(num_dimensions) * (
                                    global_best['position'] - particle['position']))
            particle['velocity'] = new_velocity.copy()
            # 更新位置
            for d in range(num_dimensions):
                particle['position'][d] += particle['velocity'][d]
            # 处理位置的边界条件
            for d in range(num_dimensions):
                if particle['position'][d] < x_min:
                    particle['position'][d] = x_min
                    particle['velocity'][d] = -particle['velocity'][d] * (1 - (iteration / num_iterations))  # 减速反弹
                elif particle['position'][d] > x_max:
                    particle['position'][d] = x_max
                    particle['velocity'][d] = -particle['velocity'][d] * (1 - (iteration / num_iterations))  # 减速反弹
            # 处理速度的边界条件
            for d in range(num_dimensions):
                if particle['velocity'][d] < v_min:
                    particle['velocity'][d] = v_min
                elif particle['velocity'][d] > v_max:
                    particle['velocity'][d] = v_max
            if not fitness_function(Group, particle['position']):
                particle['position'] = offloading_modify_method(Group, particle['position'])
        # 若结果5次没变则推出迭代
        if best_fitness == global_best['fitness']:
            no_change_count += 1
        else:
            no_change_count = 0
            best_fitness = global_best['fitness']
        if no_change_count >= max_no_change:
            break
    # 还需要判断最大能量约束
    while len(global_best_history) != 0:
        solution_fitness = fitness_function(Group, global_best_history[-1])
        if solution_fitness or solution_fitness == 0:
            print([f'{num:.5f}' for num in global_best_history[-1]])
            return global_best_history[-1], solution_fitness
        else:
            print("不满足约束，退回")
            global_best_history.pop(-1)
    print(f"全部不满足")
    return solution_0, 0

def update_tabu_list(tabu_list, delta_fitness, current_fitness, tabu_size, current_solution, neighbor_solution):
    delta_solution = [abs(a - b) for a, b in zip(neighbor_solution, current_solution)]
    suspected_task = delta_solution.index(max(delta_solution))
    if current_fitness == 0:
        tabu_list.append((suspected_task, 0))
    else:
        tabu_list.append((suspected_task, delta_fitness / current_fitness))

    if len(tabu_list) >= tabu_size:
        min_score_task = min(tabu_list, key=lambda x: x[1])
        tabu_list.remove(min_score_task)

def neighbor_generation_method(tabu_list, neighbor_solution, r, max_iteration, Group):
    """
    满意度轮盘赌，越低越容易被选择到
    """
    # 提取禁忌列表中任务的索引
    tabu_index_list = [a[0] for a in tabu_list]

    valid_indices = [i for i in range(len(neighbor_solution)) if i not in tabu_index_list]
    disturbance_index = random.choice(valid_indices)
    neighbor_solution[disturbance_index] = random.uniform(0, 1)

    aver = sum(neighbor_solution) / len(neighbor_solution)
    neighbor_solution[disturbance_index] = max(
        min(neighbor_solution[disturbance_index] + random.uniform(-aver, aver), 1), 0)

    for u, s in zip(Group.users, neighbor_solution):
        u.alpha = s
    if energy_constraint(Group.drone_k):
        return neighbor_solution
    else:
        return offloading_modify_method(Group, neighbor_solution)

def Tau_SA(Group, initial_temperature, cooling_rate, max_iteration, run_time):
    if run_time == 'before':
        current_solution = initialize_population(Group)  # 初始解，随机生成卸载比例
    elif run_time == 'later':
        current_solution = np.array([u.alpha for u in Group.users])
    else:
        raise ValueError("SA错误的运行时机")
    if isinstance(current_solution, bool):
        return np.zeros(len(Group.users), dtype=float), 0
    current_fitness = fitness_function(Group, current_solution)
    temperature = initial_temperature
    r = 0
    s = time.time()
    time_change = []
    change_range = 0.1  # 卸载率每次改变的范围
    change_time = 0  # 记录有多少次没有改变

    max_no_change = 100  # 最多100次没变
    change_fitness = current_fitness
    no_energy_time = 0
    # 定义禁忌表
    tabu_list = []
    tabu_size = math.floor(len(Group.users) * 0.25)  # 禁忌表设置为该组的用户的一半
    print("开始迭代")
    while r < max_iteration:
        r += 1
        n = 0
        neighbor_solution = current_solution.copy()
        neighbor_solution = neighbor_generation_method(tabu_list, neighbor_solution, r, max_iteration, Group)
        if isinstance(neighbor_solution, bool):
            continue
        neighbor_fitness = fitness_function(Group, neighbor_solution)
        delta_fitness = current_fitness - neighbor_fitness
        if delta_fitness < 0 or random.random() < np.exp(-(delta_fitness * len(Group.users)) / temperature):
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
        if delta_fitness > 0:
            # 如果这个更新没变化，就放到禁忌表中，更新禁忌表
            update_tabu_list(tabu_list, delta_fitness, current_fitness, tabu_size, current_solution, neighbor_solution)
        if current_fitness == change_fitness:
            no_energy_time += 1  # 计算连续几代没有找到解,
            change_time += 1
        else:
            change_time = 0
            change_fitness = current_fitness
        if change_time > max_no_change:
            print(f"总共运行次数：{no_energy_time}")
            return current_solution, current_fitness
        # 降低温度
        temperature *= cooling_rate
    print(f"2000次迭代")
    return current_solution, current_fitness

def ideal_offloading(Groups, drones):
    total = 0
    # 计算每个组用户最理想状态下的卸载率
    for group in Groups:
        group_satisfaction = []
        for drone in drones:
            group.drone_k = drone
            drone.serve_group.append(group)
            drone.Q.append((group.chloc, drone.hk))
            if not energy_constraint(drone):
                print(f"无人机{drone.k},飞不到组：{group.h}")
                solution = np.zeros(len(group.users), dtype=float)
                fitness = 0
            else:
                initial_temperature = 1000
                cooling_rate = 0.95  # 0.95
                max_iteration = 2000
                solution, fitness = Tau_SA(group, initial_temperature, cooling_rate, max_iteration, 'before')
            group_satisfaction.append((solution, fitness))
            # 退回
            group.drone_k = None
            drone.serve_group.pop(-1)
            drone.Q.pop(-1)
        group_alpha = max(group_satisfaction, key=lambda x: x[1])  # 拿出在这些无人机中，最理想状态的满意度最大的一个情况的卸载率
        group.satisfaction = group_alpha[1]  # 把最理想的值给组
        for u, s in zip(group.users, group_alpha[0]):  # 将这个卸载率给这个组内的用户，让这个组到达理想状态
            u.alpha = s
        total += group.satisfaction * len(group.users)
        print(f"{group.h}——》完成")
    print(total)
    return Groups

def compute_weight_value(K, C):
    satisfaction_score = [group.satisfaction for group in C]
    priority_score = [group.chpri for group in C]
    distance_score = []
    for group in C:
        distance = []
        for drone in K:
            # 计算组到每个无人机的距离
            distance.append(
                ((drone.Q[-1][0][0] - group.chloc[0]) ** 2 + (drone.Q[-1][0][1] - group.chloc[1]) ** 2) ** 0.5)
        # 组到无人机的平均距离
        distance_score.append(sum(distance) / len(distance))
    # 归一化
    min_p = min(priority_score)
    max_p = max(priority_score)
    min_d = min(distance_score)
    max_d = max(distance_score)
    min_s = min(satisfaction_score)
    max_s = max(satisfaction_score)
    if (max_s - min_s) == 0:
        satisfaction_score = [1 for score_s in satisfaction_score]
    else:
        satisfaction_score = [(score_s - min_s) / (max_s - min_s) for score_s in satisfaction_score]
    priority_score = [(score_p - min_p) / (max_p - min_p) for score_p in priority_score]
    distance_score = [(max_d - score_d) / (max_d - min_d) for score_d in distance_score]
    weight_priority = 0  # 优先级权重-》影响用户剩余能量与最大能量的关系从而影响满意度
    weight_distance = 0  # 距离权重-》影响时间
    weitgh_satisfaction = 1  # 满意度权重-》直接影响满意度
    composite_score = [weight_priority * p_score + weight_distance * d_score + weitgh_satisfaction * s_score for
                       p_score, d_score, s_score in
                       zip(priority_score, distance_score, satisfaction_score)]
    # 使用enumerate函数获取元素的值和索引，然后按值降序排序
    sorted_list = sorted(enumerate(composite_score), key=lambda x: x[1], reverse=True)
    # 获取前k个综合分数最大值的组
    G = [C[index] for index, value in sorted_list[:len(K)]]
    return G

def SA_path_offloading(Groups, drones):
    K = np.array(drones.copy())  # k个无人机，G与K长度相等
    C = np.array(Groups.copy())  # 所有剩余可分配的组
    t = 0
    total_time = 0
    while len(C) != 0 and len(K) != 0:
        # 将无人机计算一个理想卸载率，并将剩下的组排序
        print(f"    第{t}次调度:")
        C = ideal_offloading(C, K)
        offstime = time.time()
        t = t + 1  # 调度索引
        G = np.empty(len(K), dtype=object)  # 预分配出去的组(不考虑能量约束)
        S = []  # 本次调度分配出去的组
        if len(C) > len(K):  # 剩下的组大于无人机数量
            G = compute_weight_value(K, C)
        elif len(C) == len(K):  # 剩下的组与无人机数量相等
            G = C[:]
        else:
            G[0:len(C)] = C[0:len(C)]
            for i in range(len(C), len(K)):  # 如果最后剩余的组不够分配给无人机,则加入足够空集
                G[i] = None
        x = np.zeros([len(Groups), len(K)], dtype=int)  # 建立一个二维，(所有组，所有有足够能量的无人机)
        best_group = np.empty(len(K), dtype=object)
        if len(G) != 1:
            best_solution = simulated_annealing(G, K, x, Groups)
            total_time = total_time + (time.time() - offstime)
            for i, s in enumerate(best_solution):  # 把染色体上的数字换成对应的组对象
                if s == -1:
                    best_group[i] = None
                    continue
                for g in Groups:
                    if s == g.h:
                        best_group[i] = g
        else:
            best_solution = np.array([G[0].h])
            best_group = G.copy()
        K_delete = []
        for k, drone in enumerate(K):  # 更新无人机的服务群体组和无人机轨迹
            if best_group[k] is None:
                continue
            else:
                drone.serve_group.append(best_group[k])
                drone.Q.append((best_group[k].chloc, drone.hk))
                best_group[k].drone_k = drone
                # 验证该分配决策是否符合能量约束
                if not energy_constraint(drone):
                    print(f"不符合能量约束")
                    drone.serve_group.pop()  # 无人机不服务这个组
                    drone.Q.pop()  # 不添加轨迹
                    best_group[k].drone_k = None  # 该组不分配给此无人机
                    K_delete.append(drone)  # 无人机能量不足,不再继续给此无人机分配,得到K(t+1)
                else:
                    S.append(drone.serve_group[-1])  # 记录分配出去的组
        for s in S:  # 从组中去掉已经分配的S组得到C(t+1)
            C = C[C != s]
        for k in K_delete:  # 去掉能量不足的组，
            K = K[K != k]
        print(f"    本次调度分配出去的组：{[s.h for s in S]}")
        print(f"    所有剩余可分配的组C:{[c.h for c in C]}")
        print(f"    剩下有足够能量的无人机K(t):{[a.k for a in K]}")
    for d in drones:
        print(f"{d.k}所服务的组")
        for g in d.serve_group:
            print(g.h)
    print(total_satisfaction(Groups))
    return total_satisfaction(Groups), Groups, drones, total_time
