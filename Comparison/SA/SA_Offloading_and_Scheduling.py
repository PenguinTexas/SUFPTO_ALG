import sys
from GA_based_UAV_Scheduling_Algorithm import GA_based_UAV_Scheduling_Algorithm
from Simulated_Annealing import simulated_annealing


# 尝试ga+sa
def Joint_scheduling_offloading(users, drones, Groups, assign, scheduling, offloading):
    drones, Groups = GA_based_UAV_Scheduling_Algorithm(drones, Groups, assign, scheduling)

    if offloading == 'SA':
        initial_temperature = 1000
        cooling_rate = 0.998  # 0.995
        max_iteration = 20000
        satisfaction, Groups, drones = simulated_annealing(users, drones, Groups, initial_temperature, cooling_rate,
                                                           max_iteration)
    else:
        print("任务卸载没有指定算法")
        sys.exit()
    return satisfaction, Groups, drones


def Joint_Task_Offloading_and_UAV_Scheduling_Optimization_Algorithm():
    pass
