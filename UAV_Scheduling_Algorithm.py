import math
import random
import numpy as np
from model_compute import calculate_user_satisfaction, E_back, E_i_k
import sys
from model_compute import energy_constraint

# 模拟退火算法_scheduling
def simulated_annealing(G, K, x, Groups):
    # 1. 初始化当前解：将组ID映射到解向量（未分配组设为-1）
    current_solution = np.array([g.h if g is not None else -1 for g in G], dtype=int)
    current_cost = fitness_function(current_solution, K, x, Groups)
    # 2. 模拟退火参数
    temperature = 1000
    cooling_rate = 0.95
    max_iterations = 1000
    # 3. 退火迭代
    for t in range(max_iterations):
        # 3.1 生成新解：随机交换两个组的分配
        new_solution = current_solution.copy()
        i, j = random.sample(range(len(G)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        # 3.2 计算新解成本
        new_cost = fitness_function(new_solution, K, x, Groups)
        cost_diff = new_cost - current_cost
        # 3.3 接受准则：更优解或概率接受劣解
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_solution, current_cost = new_solution, new_cost
        # 3.4 降温
        temperature *= cooling_rate
    return current_solution

# 计算当无人机k服务用户组g时的成本
def u_gk(group, drone):  #
    # 1. 临时分配：假设无人机服务该组
    drone.serve_group.append(group)
    group.drone_k = drone
    # 2. 计算组内用户总满意度
    total_satisfaction = sum(
        calculate_user_satisfaction(j, group)
        for j in range(len(group.users))
    )
    # 3. 撤销临时分配
    drone.serve_group.pop()
    group.drone_k = None
    # 4. 返回成本的负值
    return -total_satisfaction

# 计算一个调度方案的总成本
def calculate_cost(solution, drones, assignment_matrix, all_groups):
    total_cost = 0
    # 将解向量转换为组对象列表
    assigned_groups = []
    for group_id in solution:
        if group_id == -1 or group_id >= 10000:  # 无效组ID
            assigned_groups.append(None)
            continue
        # 查找对应的组对象
        group = next((g for g in all_groups if g.h == group_id), None)
        assigned_groups.append(group)
    # 计算每个组的成本
    for group, drone in zip(assigned_groups, drones):
        if group is None:  # 未分配组，成本为0
            continue
        # 检查分配矩阵，找到负责该组的无人机
        for drone_idx, drone_obj in enumerate(drones):
            if assignment_matrix[group.h][drone_idx] == 1:
                total_cost += u_gk(group, drone_obj)
                break
    return total_cost

# 适应度函数
def fitness_function(solution, drones, assignment_matrix, Groups):
    total_cost = 0
    for i, k in zip(solution, range(len(drones))):
        if i >= 10000 or i == -1:  # 说明是空集
            continue
        else:
            assignment_matrix[i, k] = 1
    # 计算成本
    cost = calculate_cost(solution, drones, assignment_matrix, Groups)
    total_cost += cost
    for i, k in zip(solution, range(len(drones))):  # 计算完成后归零，以便下次计算
        if i >= 10000 or i == -1:  # 说明是空集
            continue
        else:
            assignment_matrix[i, k] = 0
    return total_cost

# 初始化种群
def initialize_population(pop_size, groups):
    population = []
    for _ in range(pop_size):
        # 创建解向量
        solution = np.array([g.h if g is not None else -1 for g in groups])
        # 随机打乱两次
        np.random.shuffle(solution)
        np.random.shuffle(solution)
        population.append(solution)
    return np.array(population, dtype=object)

# 自我优化操作
def self_optimization(index, elite_pop, drones, assign_mat, groups):
    elite = elite_pop[index].copy()  # 复制精英个体
    # 随机交换两个基因
    i, j = random.sample(range(len(elite)), 2)
    elite[i], elite[j] = elite[j], elite[i]
    # 保留更优解
    if fitness_function(elite, drones, assign_mat, groups) < \
            fitness_function(elite_pop[index], drones, assign_mat, groups):
        elite_pop[index] = elite
    return elite_pop

# 交叉操作 - PMX 交叉
def pmx_crossover(p1, p2):
    parent1 = p1.copy()
    parent2 = p2.copy()
    d = 10000
    for a in range(len(parent1)):
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
    for a in range(len(child1)):
        if child1[a] >= 10000:
            child1[a] = -1
        if child2[a] >= 10000:
            child2[a] = -1
    return child1, child2

# 突变操作 - 基因交换
def mutation(individual, *args):
    mutant = individual.copy()
    i, j = random.sample(range(len(mutant)), 2)
    mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant

# GA_for_Balanced_Assignment_Problem
def GA_for_Balanced_Assignment_Problem(G, K, x, Groups):
    # 参数配置
    population_size = 20  # 种群大小20
    Niter1 = 30  # 迭代次数
    crossover_rate = 0.8  # 交叉率pc1
    mutation_rate = 0.06  # 变异率pm1
    Ns1 = 10  # 自优化的基因交换次数
    elite_size = 1  # 保留的精英个体数量
    # 初始化种群
    population = initialize_population(population_size, G)
    # 主循环：遗传算法的迭代过程
    for r in range(Niter1):
        print(f"        第 {r} 次迭代................................................................")
        # 评估并排序种群
        population = sorted(population, key=lambda y: fitness_function(y, K, x, Groups))
        # 精英保留与自我优化
        elite = population[:elite_size]
        for i in range(elite_size):
            for _ in range(Ns1):
                elite = self_optimization(i, elite, K, x, Groups)
        # 将优化后的精英个体添加到新种群，生成新种群
        new_population = np.empty(population_size, dtype=object)
        new_population[0] = elite[0]
        new_population_length = 1
        temp_population = np.array(population[elite_size:])
        temp_fitness = np.array([fitness_function(solution, K, x, Groups) for solution in temp_population])
        if sum(temp_fitness) == 0:  # 可能意味着无人机太少，最后分配的时间过长导致满意度为零
            Pr = np.full(len(temp_fitness), 1 / len(temp_fitness))
        else:
            Pr = np.array([fitness / sum(temp_fitness) for fitness in temp_fitness])
        while new_population_length < population_size:
            parent1 = random.choices(temp_population, weights=Pr, k=1)[0]
            parent2 = elite[0]
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

def GA_based_UAV_Scheduling_Algorithm(drones, Groups, assign, scheduling):
    print(f"    无人机调度优化.............")
    # 初始化无人机和组状态
    for drone in drones:
        drone.serve_group = []
    for group in Groups:
        group.drone_k = None
    K = np.array(drones)
    C = np.array(Groups)
    t = 0
    while len(C) > 0 and len(K) > 0:
        print(f"第{t}次调度:")
        t = t + 1  # 调度索引
        G = np.empty(len(K), dtype=object)  # 预分配出去的组
        # 1. 选择待分配组G
        if len(C) > len(K):  # 剩下的组大于无人机数量
            if assign == 'p':
                G = C[0:len(K)]  # 最优的k(t)个组，即K个最优先的组
            elif assign == 'p82':
                G = assignment(C, K, 0.8)
            elif assign == 'p55':
                G = assignment(C, K, 0.5)
            else:
                print("分配的参数传输错误")
                sys.exit()
        elif len(C) == len(K):  # 剩下的组与无人机数量相等
            G = C[:]
        else:
            G[0:len(C)] = C[0:len(C)]
            for i in range(len(C), len(K)):  # 如果最后剩余的组不够分配给无人机,则加入足够空集
                G[i] = None
        # 2. 运行调度算法
        x = np.zeros((len(Groups), len(K)), dtype=int)  # 建立一个二维，(所有组，所有有足够能量的无人机)
        best_group = np.empty(len(K), dtype=object)
        if len(G) > 1:
            # 两种算法
            if scheduling == 'GA':
                best_solution = min(GA_for_Balanced_Assignment_Problem(G, K, x, Groups),
                                    key=lambda y: fitness_function(y, K, x, Groups))
            elif scheduling == 'SA':
                best_solution = simulated_annealing(G, K, x, Groups)
            else:
                print("调度的算法名称错误")
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
        # 3. 分配组并检查能量约束
        S, K_delete = [], []
        for drone, group in zip(K, best_group):     # 更新无人机的服务群体组和无人机轨迹
            if not group:
                continue
            drone.serve_group.append(group)
            drone.Q.append((group.chloc, drone.hk))
            group.drone_k = drone
            # 验证该分配决策是否符合能量约束
            if not energy_constraint(drone):
                drone.serve_group.pop()
                drone.Q.pop()
                group.drone_k = None
                K_delete.append(drone)
            else:
                S.append(group)
        # 4. 更新剩余组和无人机
        C = np.array([g for g in C if g not in S])
        K = np.array([k for k in K if k not in K_delete])
    return drones, Groups
