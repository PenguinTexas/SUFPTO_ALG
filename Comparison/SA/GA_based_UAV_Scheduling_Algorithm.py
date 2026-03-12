import math
import random
import time

import numpy as np

from model_compute import calculate_user_satisfaction, E_back, E_i_k
import sys


# 模拟退火算法_scheduling
def simulated_annealing(G, K, x, Groups):
    solution = np.zeros(len(G), dtype=int)
    for j, g in enumerate(G):
        if g is not None:
            solution[j] = g.h
        else:
            solution[j] = -1  # 如果是空集就用-1放入染色体

    current_solution = solution.copy()
    current_cost = fitness_function(current_solution, K, x, Groups)
    temperature = 100
    cooling_rate = 0.95
    fitness_history = []  # 用于存储适应度随时间的变化
    t = 0
    iteration = 500
    diff = []
    while t < iteration:
        t += 1
        # 生成新的排列
        new_solution = current_solution.copy()
        i, j = random.sample(range(len(current_solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_cost = fitness_function(new_solution, K, x, Groups)

        # 计算成本差异
        cost_diff = new_cost - current_cost
        diff.append(cost_diff)
        # 如果新解更好或者以一定概率接受差解
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_solution = new_solution
            current_cost = new_cost
            if cost_diff != 0:
                fitness_history.append([f"第{t}次：", current_cost, current_solution])
        # 降低温度
        temperature *= cooling_rate
    return current_solution


# 如果组Gg由UAV k服务，则有一个成本定义定义成本
def u_gk(g, k):
    k.serve_group.append(g)  # 假设无人机k服务多一组g
    g.drone_k = k  # 假设组g由无人机k服务
    u_total = 0  # 初始满意度
    for j, user in enumerate(g.users):
        u_total += calculate_user_satisfaction(j, g)
    k.serve_group.pop()  # 撤销假设
    g.drone_k = None
    return -u_total


# 计算成本
def calculate_cost(solution, K, x, Groups):
    cost = 0
    groups = np.empty(len(solution), dtype=object)
    for i, s in enumerate(solution):  # 把染色体上的数字换成对应的组对象
        if s == -1 or s >= 10000:
            groups[i] = None
            continue
        for g in Groups:
            if s == g.h:
                groups[i] = g
                break

    for group in groups:
        if group is None:  # 如果当前无人机没有分配组,则成本为零
            cost += 0
            continue
        for k, k_uav in enumerate(K):
            if x[group.h][k] == 1:
                cost += x[group.h][k] * u_gk(group, k_uav)
                break
            else:
                continue

    return cost


# 适应度函数
def fitness_function(solution, K, x, Groups):  # solution=(3,7,9),K=(0,1,2)
    total_cost = 0

    for i, k in zip(solution, range(len(K))):
        if i >= 10000 or i == -1:  # 说明是空集
            continue
        else:
            x[i, k] = 1

    # 计算成本
    cost = calculate_cost(solution, K, x, Groups)
    total_cost += cost

    for i, k in zip(solution, range(len(K))):  # 计算完成后归零，以便下次计算
        if i >= 10000 or i == -1:  # 说明是空集
            continue
        else:
            x[i, k] = 0
    return total_cost


# 初始化种群
def initialize_population(population_size, G):
    population_0 = np.empty(population_size, dtype=object)

    for i in range(population_size):
        solution = np.zeros(len(G), dtype=int)
        for j, g in enumerate(G):
            if g is not None:
                solution[j] = g.h
            else:
                solution[j] = -1  # 如果是空集就用-1放入染色体

        np.random.shuffle(solution)
        np.random.shuffle(solution)
        population_0[i] = solution
    return population_0


# 自我优化操作
def self_optimization(i, elite, K, x, Groups):
    # 复制精英染色体以进行交换
    new_elite = elite[i][:]  # 复制精英染色体
    # 随机选择两个位置进行基因交换
    idx1, idx2 = random.sample(range(len(new_elite)), 2)
    new_elite[idx1], new_elite[idx2] = new_elite[idx2], new_elite[idx1]

    # 计算交换后的适应度
    cost_after_swap = fitness_function(new_elite, K, x, Groups)

    # 如果交换后的适应度更好，则保留交换结果
    if cost_after_swap < fitness_function(elite[i], K, x, Groups):
        elite[i] = new_elite  # 更新精英染色体
    return elite


# 交叉操作 - PMX 交叉
def pmx_crossover(p1, p2):
    parent1 = p1.copy()
    parent2 = p2.copy()
    d = 10000
    for a in range(len(parent1)):  # 在最后一次调度时,种群内的多个空集为-1，换成不同数，从而进行Pmx交叉
        if parent1[a] == -1:
            parent1[a] = d
            d += 1
    e = 10000
    for b in range(len(parent2)):
        if parent2[b] == -1:
            parent2[b] = e
            e += 1
    # 随机选择两个交叉点
    start, end = sorted(random.sample(range(len(parent1)), 2))

    # 创建两个子代
    child1 = np.full(len(parent1), -2)
    child2 = np.full(len(parent2), -2)

    # 复制父代的交叉点之间的基因到子代
    child1[start:end + 1] = parent1[start:end + 1]
    child2[start:end + 1] = parent2[start:end + 1]

    # 处理交叉点之外的基因
    for i in range(len(parent1)):
        if child1[i] == -2:
            corresponding_gene = parent2[i]
            while corresponding_gene in child1[start:end + 1]:
                corresponding_gene = parent2[np.where(parent1 == corresponding_gene)]
            child1[i] = corresponding_gene

        if child2[i] == -2:
            corresponding_gene = parent1[i]
            while corresponding_gene in child2[start:end + 1]:
                corresponding_gene = parent1[np.where(parent2 == corresponding_gene)]
            child2[i] = corresponding_gene
    for a in range(len(child1)):  # 在最后一次调度时,种群内的多个空集为-1，换成不同数，从而进行Pmx交叉
        if child1[a] >= 10000:
            child1[a] = -1
        if child2[a] >= 10000:
            child2[a] = -1
    return child1, child2


# 突变操作 - 基因交换
def mutation(child, K, x, Groups):
    new_child = child.copy()
    # 随机选择两个不同的基因位置
    idx1, idx2 = random.sample(range(len(new_child)), 2)

    # 交换两个基因的值
    new_child[idx1], new_child[idx2] = new_child[idx2], new_child[idx1]
    return new_child


# GA_for_Balanced_Assignment_Problem
def GA_for_Balanced_Assignment_Problem(G, K, x, Groups):
    # 种群大小L1设置为20。迭代次数Niter1设置为30。交叉率Pc1和变异率pm1分别设置为0.8和0.06。自优化次数Ns1设置为10。
    # 先初始化种群
    population_size = 20  # 种群大小20
    population = initialize_population(population_size, G)

    # 定义遗传算法参数
    Niter1 = 30  # 迭代次数
    crossover_rate = 0.8  # 交叉率pc1
    mutation_rate = 0.06  # 变异率pm1
    Ns1 = 10  # 自优化的基因交换次数
    # 主循环：遗传算法的迭代过程
    for r in range(Niter1):
        print(f"        第 {r} 次迭代................................................................")
        # print(f"        执行遗传算法")
        ga_time = time.time()
        # 计算适应度并排序
        population = sorted(population, key=lambda y: fitness_function(y, K, x, Groups))
        # 选择最优解
        elite_size = 1  # 保留的精英个体数量
        elite = population[:elite_size]

        # 自我优化操作
        for i in range(elite_size):
            for _ in range(Ns1):
                # print("执行自我优化操作")
                elite = self_optimization(i, elite, K, x, Groups)

        # 将优化后的精英个体添加到新种群
        # 生成新种群
        new_population = np.empty(population_size, dtype=object)
        new_population[0] = elite[0]
        new_population_length = 1
        temp_population = np.array(population[elite_size:])
        temp_fitness = np.array([fitness_function(solution, K, x, Groups) for solution in temp_population])
        if sum(temp_fitness) == 0:  # 可能意味着无人机太少，最后分配的时间过长导致满意度为零
            Pr = np.full(len(temp_fitness), 1 / len(temp_fitness))
        else:
            Pr = np.array([fitness / sum(temp_fitness) for fitness in temp_fitness])
        # print("得到去掉已选择的父代染色体的种群")
        while new_population_length < population_size:
            parent1 = random.choices(temp_population, weights=Pr, k=1)[0]
            parent2 = elite[0]  # 精英现在只指定保留一个
            child = []
            if random.random() <= crossover_rate:
                child1, child2 = pmx_crossover(parent1, parent2)
            else:  # 不执行交叉保留父代
                child1, child2 = parent1, parent2
            if random.random() <= mutation_rate:
                child1 = mutation(child1, K, x, Groups)
                child2 = mutation(child2, K, x, Groups)
            c1 = fitness_function(child1, K, x, Groups)
            c2 = fitness_function(child2, K, x, Groups)
            if c1 < c2:
                child = child1
            else:
                child = child2
            new_population[new_population_length] = child
            new_population_length += 1
        population = new_population
    return population


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


# 第t次调度的k个分组的选择
def assignment(C, K, weight):  # 组,无人机
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

    priority_score = [(score_p - min_p) / (max_p - min_p) for score_p in priority_score]
    distance_score = [(max_d - score_d) / (max_d - min_d) for score_d in distance_score]
    weight_priority = weight  # 优先级权重
    weight_distance = 1 - weight  # 距离权重
    composite_score = [weight_priority * p_score + weight_distance * d_score for p_score, d_score in
                       zip(priority_score, distance_score)]

    # 使用enumerate函数获取元素的值和索引，然后按值降序排序
    sorted_list = sorted(enumerate(composite_score), key=lambda x: x[1], reverse=True)

    # 获取前k个综合分数最大值的组
    G = [C[index] for index, value in sorted_list[:len(K)]]
    return G


# 执行GA_based_UAV_Scheduling_Algorithm
def GA_based_UAV_Scheduling_Algorithm(drones, Groups, assign, scheduling):
    print(f"    无人机调度优化.............")
    for drone in drones:
        drone.serve_group = ((0, 0), drone.hk)
        drone.serve_group = []
    for Group in Groups:
        Group.drone_k = None

    K = np.array(drones.copy())  # k个无人机，G与K长度相等
    C = np.array(Groups.copy())  # 所有剩余可分配的组

    t = 0
    while len(C) != 0 and len(K) != 0:
        print(f"    第{t}次调度:")
        t = t + 1  # 调度索引
        # T = [i for i in range(math.ceil(len(C) / len(K)))]  # 调度索引

        G = np.empty(len(K), dtype=object)  # 预分配出去的组(不考虑能量约束)
        S = []  # 本次调度分配出去的组

        if len(C) > len(K):  # 剩下的组大于无人机数量
            if assign == 'p':
                G = C[0:len(K)]  # 最优的k(t)个组，即K个最优先的组
            # 按照优先级和距离同时考虑分配出去的组
            elif assign == 'p82':
                G = assignment(C, K, 0.8)
            elif assign == 'p55':
                G = assignment(C, K, 0.5)
            else:
                print("分配的参数传输错误，退出程序")
                sys.exit()
        elif len(C) == len(K):  # 剩下的组与无人机数量相等
            G = C[:]
        else:
            G[0:len(C)] = C[0:len(C)]
            for i in range(len(C), len(K)):  # 如果最后剩余的组不够分配给无人机,则加入足够空集
                G[i] = None
        # print(f"选择的k(t)个优先的组G:{[c if c is None else c.h for c in G]}")
        x = np.zeros([len(Groups), len(K)], dtype=int)  # 建立一个二维，(所有组，所有有足够能量的无人机)
        best_group = np.empty(len(K), dtype=object)
        if len(G) != 1:
            # 两种算法
            if scheduling == 'GA':
                best_solution = min(GA_for_Balanced_Assignment_Problem(G, K, x, Groups),
                                    key=lambda y: fitness_function(y, K, x, Groups))
            elif scheduling == 'SA':
                best_solution = simulated_annealing(G, K, x, Groups)
            else:
                print("调度的算法名称错误，退出")
                sys.exit()
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
                    drone.serve_group.pop()
                    drone.Q.pop()
                    best_group[k].drone_k = None  # 该组不分配给此无人机
                    K_delete.append(drone)  # 无人机能量不足,不再继续给此无人机分配,得到K(t+1)
                else:
                    S.append(drone.serve_group[-1])  # 记录分配出去的组
        for s in S:  # 从组中去掉已经分配的S组得到C(t+1)
            C = C[C != s]
        for k in K_delete:
            K = K[K != k]
    return drones, Groups
