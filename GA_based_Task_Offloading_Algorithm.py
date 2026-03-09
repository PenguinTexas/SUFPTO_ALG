import random
import time
import numpy as np
from model_compute import energy_total_constraint, calculate_user_satisfaction

max_attempts = 100  # 设置循环的最大尝试次数，能量约束处
# 对一组有无人机服务的用户群，生成满足能量约束的候选解（卸载比例）
def initialize_population(population_size, users, drones):
    population = []
    for _ in range(population_size):
        for _ in range(max_attempts):
            # 生成随机解
            solution = np.random.uniform(0, 1, len(users))
            # 设置用户alpha值
            for u, alpha in zip(users, solution):
                u.alpha = alpha
            # 检查约束
            if energy_total_constraint(drones):
                population.append(solution)
                break
        else:  # 尝试max_attempts次后仍未满足约束
            zero_solution = np.zeros(len(users))
            population.append(zero_solution)
            # 设置全零解
            for u in users:
                u.alpha = 0
    return np.array(population, dtype=object)

# 根据用户满意度模型生成适应度函数
def fitness_function(solution, users, drones, Groups):
    total_cost = 0
    for u, s in zip(users, solution):
        u.alpha = s
    if not energy_total_constraint(drones):
        return False
    for g in Groups:
        for j in range(len(g.users)):
            total_cost += calculate_user_satisfaction(j, g)
    total_cost = round(total_cost, 5)
    return total_cost

# 能量约束测试
def solution_constraint(solution, users, drones):
    for u, s in zip(users, solution):
        u.alpha = s
    if not energy_total_constraint(drones):
        return False
    else:
        return True

# 爬山算法
def hill_climbing(Ns2, initial_solution, users, drones, Groups):
    current_solution = initial_solution.copy()
    current_fitness = fitness_function(current_solution, users, drones, Groups)
    for _ in range(Ns2):
        # 生成可行邻域解
        for _ in range(max_attempts):
            neighbor = current_solution.copy()
            idx = random.randint(0, len(neighbor) - 1)
            delta = random.uniform(-0.1, 0.1)
            # 边界检查
            if neighbor[idx] + delta > 1:
                delta = random.uniform(-0.1, 1 - neighbor[idx])
            elif neighbor[idx] + delta < 0:
                delta = random.uniform(-neighbor[idx], 0.1)
            neighbor[idx] += delta
            if solution_constraint(neighbor, users, drones):
                break
        else:  # 超过最大尝试次数
            neighbor = initial_solution.copy()
        # 评估并接受更好的解
        neighbor_fitness = fitness_function(neighbor, users, drones, Groups)
        if neighbor_fitness > current_fitness:
            current_solution = neighbor
    return current_solution

# 选择操作
def selection(population, Pr):
    return random.choices(population, weights=Pr, k=1)[0]

# 交叉操作
def crossover(parent1, parent2, users, drones):
    for _ in range(max_attempts):
        # 创建父代副本
        child1, child2 = parent1.copy(), parent2.copy()
        # 两点交叉
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child1[start:end], child2[start:end] = child2[start:end], child1[start:end]
        # 检查约束
        valid = []
        if solution_constraint(child1, users, drones):
            valid.append(child1)
        if solution_constraint(child2, users, drones):
            valid.append(child2)
        if valid:  # 如果有可行解
            return random.choice(valid)  # 随机返回一个可行子代
    return parent1  # 超过尝试次数返回父代1

# 突变操作
def mutate(child, Nm2, users, drones):
    for _ in range(max_attempts):
        mutant = child.copy()
        # 执行Nm2次变异
        for _ in range(Nm2):
            idx = random.randint(0, len(mutant) - 1)
            # 计算安全的变异范围
            lower = max(-0.1, -mutant[idx])
            upper = min(0.1, 1 - mutant[idx])
            mutant[idx] += random.uniform(lower, upper)
        if solution_constraint(mutant, users, drones):
            return mutant
    return child  # 超过尝试次数返回原解

# 基于遗传算法改进任务卸载比例
def GA_based_Task_Offloading_Algorithm(users, drones, Groups, population_size, Niter2, crossover_rate, mutation_rate, Ns2, Nm2):
    last_alpha = [u.alpha for u in users]  # 记录上次迭代结果
    print("任务卸载优化开始...")
    # 初始化种群
    population = initialize_population(population_size, users, drones)
    # 主循环：遗传算法的迭代过程
    for iteration in range(Niter2):  # 迭代索引
        start_time_t = time.time()
        print(f"第 {iteration} 次迭代...")
        # 计算每个解的适应度并存储在列表中
        population_with_fitness = [
            (individual, fitness_function(individual, users, drones, Groups))
            for individual in population
        ]
        # 按适应度排序
        population_sorted = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
        population = np.array([individual for individual, _ in population_sorted])  # 排序后种群
        fitness_values = np.array([fitness for _, fitness in population_sorted])    # 排序后的适应度
        print(f"当前最高适应度：{fitness_values[0]}")
        # 精英保留策略
        elite = [hill_climbing(Ns2, population[0], users, drones, Groups)]
        # 生成新一代种群
        new_population = [elite[0]]  # 保留精英
        # 计算选择概率（排除精英）
        remaining_pop = population[1:]
        remaining_fitness = fitness_values[1:]
        selection_probs = remaining_fitness / remaining_fitness.sum()
        # 填充剩余种群
        while len(new_population) < population_size:
            # 选择父代
            parent1, parent2 = (
                selection(remaining_pop, selection_probs),
                selection(remaining_pop, selection_probs)
            )
            # 交叉
            child = (crossover(parent1, parent2, users, drones)
                     if random.random() < crossover_rate else parent1)
            # 变异
            if random.random() < mutation_rate:
                child = mutate(child, Nm2, users, drones)
            new_population.append(child)
        population = np.array(new_population, dtype=object)
        end_time_t = time.time()
        print(f"迭代耗时：{end_time_t - start_time_t:.2f}秒")
    # 选择最终最优解
    final_population = population.tolist() + [last_alpha]
    best_solution = max(final_population, key=lambda x: fitness_function(x, users, drones, Groups))
    # 替换成最优的任务卸载比列
    for user, alpha in zip(users, best_solution):
        user.alpha = alpha
    return users, drones, Groups
