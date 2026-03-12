import random
import time
import numpy as np
from GA_based_Task_Offloading_Algorithm import fitness_function, initialize_population


# 模拟退火算法函数,卸载比例
def simulated_annealing(users, drones, Groups, initial_temperature, cooling_rate, max_iteration):
    current_solution = initialize_population(1, users, drones)[0]  # 初始解，随机生成卸载比例
    current_fitness = fitness_function(current_solution, users, drones, Groups)

    temperature = initial_temperature
    fitness_history = []  # 用于存储适应度随时间的变化
    r = 0
    s = time.time()
    time_change = []

    change_time = 0  # 记录有多少次没有改变
    change_fitness = current_fitness
    max_no_change = 500  # 最多100次没变
    while r < max_iteration:
        r += 1
        # print("开始退火")
        print(f"annealing time:{r}，run_time:{time.time() - s},current_fitness:{current_fitness}")
        # print(f"temperature:{temperature}")

        n = 0

        neighbor_solution = current_solution.copy()
        while True:

            random_index = random.randint(0, len(neighbor_solution) - 1)
            if neighbor_solution[random_index] + 0.1 > 1:
                disturbance_value = random.uniform(-0.1, 1 - neighbor_solution[random_index])
                # disturbance_value = random.uniform(-0.1, 0)
            elif neighbor_solution[random_index] - 0.1 < 0:
                disturbance_value = random.uniform(0 - neighbor_solution[random_index], 0.1)
                # disturbance_value = random.uniform(0, 0.1)
            else:
                disturbance_value = random.uniform(-0.1, 0.1)
            neighbor_solution[random_index] += disturbance_value
            neighbor_fitness = fitness_function(neighbor_solution, users, drones, Groups)
            if neighbor_fitness:
                break
            else:
                neighbor_solution[random_index] -= disturbance_value
                n += 1
                if n > 10:  # 结束无限循环
                    break

        delta_fitness = current_fitness - neighbor_fitness

        if delta_fitness < 0 or random.random() < np.exp(-delta_fitness / temperature):
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
            fitness_history.append(current_fitness)  # 用于存储适应度随时间的变化
            if delta_fitness < -0.005:
                time_change.append([r, current_fitness])
        if current_fitness == change_fitness:
            change_time += 1
            # print(f"没有变次数加一：{change_time}")
        else:
            change_time = 0
            change_fitness = current_fitness
        if change_time > max_no_change:
            return current_fitness, Groups, drones
        # 降低温度
        temperature *= cooling_rate

    return current_fitness, Groups, drones
