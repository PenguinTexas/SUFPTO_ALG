import random
import time
import numpy as np
from GA_based_UAV_Scheduling_Algorithm import energy_constraint
from model_compute import calculate_user_satisfaction

max_attempts = 100  # 设置循环的最大尝试次数，能量约束处


def energy_total_constraint(drones):
    for drone in drones:
        if not energy_constraint(drone):
            return False
    return True


def initialize_population(population_size, users, drones):
    population_0 = np.empty(population_size, dtype=object)  # 初始化种群空间
    for i in range(population_size):
        attempts = 0
        while True:
            solution = np.zeros(len(users), dtype=float)
            for j, u in enumerate(users):
                random_alpha = random.uniform(0, 1)
                u.alpha = random_alpha  # 重新随机一个
                solution[j] = random_alpha
            if energy_total_constraint(drones):  # 判断满足无人机最大能量约束
                population_0[i] = solution
                break
            else:
                attempts += 1
            if attempts == max_attempts:
                # 在达到最大尝试次数后仍未满足约束，采取适当的措施，生成一个默认全零解，防止无限循环
                population_0[i] = np.zeros(len(users), dtype=float)
                break

    return population_0


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


def solution_constraint(solution, users, drones):
    for u, s in zip(users, solution):
        u.alpha = s
    if not energy_total_constraint(drones):
        return False
    else:
        return True


def hill_climbing(Ns2, initial_solution, users, drones, Groups):
    current_solution = initial_solution.copy()
    for i in range(Ns2):
        # 计算当前解的适应度（目标函数值）
        current_fitness = fitness_function(current_solution, users, drones, Groups)
        n = 0
        attempts = 0
        while True:  # 判断满足无人机最大能量约束
            n += 1
            # 生成一个邻近的解，稍微改变当前解
            neighbor_solution = current_solution.copy()
            index_to_change = random.randint(0, len(neighbor_solution) - 1)
            if neighbor_solution[index_to_change] + 0.1 > 1:
                neighbor_solution[index_to_change] += random.uniform(-0.1, 0)
            elif neighbor_solution[index_to_change] - 0.1 < 0:
                neighbor_solution[index_to_change] += random.uniform(0, 0.1)
            else:
                neighbor_solution[index_to_change] += random.uniform(-0.1, 0.1)

            if solution_constraint(neighbor_solution, users, drones):  # 判断满足最大能量约束
                break
            else:
                attempts += 1
            if attempts == max_attempts:
                # 在达到最大尝试次数后仍未满足约束，采取适当的措施，生成一个默认初始解，防止无限循环
                neighbor_solution = initial_solution
                break
        # 计算邻近解的适应度
        neighbor_fitness = fitness_function(neighbor_solution, users, drones, Groups)
        # 如果邻近解更好，则接受它作为新的当前解
        if neighbor_fitness > current_fitness:
            current_solution = neighbor_solution
        end_time_a = time.time()
    end_time = time.time()
    return current_solution


def selection(population, Pr):
    return random.choices(population, weights=Pr, k=1)[0]


# 交叉操作
def crossover(parent1, parent2, users, drones):
    start_time = time.time()
    attempts = 0
    while True:
        p1 = parent1.copy()
        p2 = parent2.copy()
        # 随机选择两个交叉点
        start, end = sorted(random.sample(range(len(p1)), 2))
        # 从父代中获取相同位置的一系列基因
        p1[start:end], p2[start:end] = p2[start:end], p1[start:end]
        p1_s = solution_constraint(p1, users, drones)
        p2_s = solution_constraint(p2, users, drones)
        if p1_s is not False and p2_s is not False:
            return p1 if random.random() < 0.5 else p2
        elif p1_s:
            return p1
        elif p2_s:
            return p2
        else:
            attempts += 1
        if attempts == max_attempts:
            # 在达到最大尝试次数后仍未满足约束，采取适当的措施，返回一个默认初始解，防止无限循环
            return parent1


# 突变操作
def mutate(child, Nm2, users, drones):
    start_time = time.time()
    attempts = 0
    while True:
        child_0 = child.copy()
        for _ in range(Nm2):
            # 随机选择要突变的基因位置
            mutation_position = random.randint(0, len(child_0) - 1)
            # 随机生成一个增加或减少的随机数
            if child_0[mutation_position] + 0.1 > 1:
                mutation_value = random.uniform(-0.1, 0)
            elif child_0[mutation_position] - 0.1 < 0:
                mutation_value = random.uniform(0, 0.1)
            else:
                mutation_value = random.uniform(-0.1, 0.1)
            # 将突变值添加到选定位置的基因上
            child_0[mutation_position] += mutation_value

        if solution_constraint(child_0, users, drones):  # 判断满足无人机最大能量约束
            end_time = time.time()
            return child_0
        else:
            attempts += 1
        if attempts == max_attempts:
            # 在达到最大尝试次数后仍未满足约束，采取适当的措施，生成一个默认初始解，防止无限循环
            return child


def GA_based_Task_Offloading_Algorithm(users, drones, Groups, population_size, Niter2, crossover_rate, mutation_rate,
                                       Ns2, Nm2):
    last_alpha = [u.alpha for u in users]  # 记录上次迭代结果
    print(f"    任务卸载优化.............")
    population = initialize_population(population_size, users, drones)
    # 主循环：遗传算法的迭代过程
    for r in range(Niter2):  # 迭代索引
        start_time_t = time.time()
        print(f"        第 {r} 次迭代................................................................:")
        individual_fitness_list = []
        # 计算每个个体的适应度并存储在列表中
        for individual in population:
            fitness = fitness_function(individual, users, drones, Groups)
            individual_fitness_list.append((individual, fitness))
        # 排序
        population_1 = sorted(individual_fitness_list, key=lambda x: x[1], reverse=True)
        # 排序后种群
        population = np.array([np.array(individual[0]) for individual in population_1])
        # 排序后的适应度
        fit_population = np.array([np.array(fit[1]) for fit in population_1])
        print(f"此次适应度最高：{fit_population[0]}")
        # 选择适应度最高的一个染色体
        elite_size = 1  # 保留的精英个体数量
        elite = population[:elite_size]
        # 对精英个体应用爬山算法来进一步优化它们
        for i in range(elite_size):
            elite[i] = hill_climbing(Ns2, elite[i], users, drones, Groups)
        # 将优化后的精英个体添加到新种群
        # 生成新种群
        new_population = np.empty(population_size, dtype=object)
        new_population[0] = elite[0]
        new_population_length = 1
        # 从不包括精英的初始种群中选择的父代
        temp_population = np.array(population[1:])
        temp_fitness = np.array(fit_population[1:])
        Pr = np.array([fitness / sum(temp_fitness) for fitness in temp_fitness])
        while new_population_length < population_size:
            # 轮盘赌选两个父代
            parent1 = selection(temp_population, Pr)
            parent2 = selection(temp_population, Pr)
            if random.random() <= crossover_rate:
                child = crossover(parent1, parent2, users, drones)
            else:  # 不执行交叉，保留父代
                child = parent1
            # 变异操作
            if random.random() <= mutation_rate:
                child = mutate(child, Nm2, users, drones)
            new_population[new_population_length] = child
            new_population_length += 1
        population = new_population
        end_time_t = time.time()

        print(f"任务卸载迭代优化一次的运行时间为:{end_time_t - start_time_t}")
    best_solution = max(population, key=lambda y: fitness_function(y, users, drones, Groups))
    this_best = max([best_solution, last_alpha], key=lambda y: fitness_function(y, users, drones, Groups))
    for user, s in zip(users, this_best):  # 替换成最优的任务卸载比列
        user.alpha = s
    return users, drones, Groups
