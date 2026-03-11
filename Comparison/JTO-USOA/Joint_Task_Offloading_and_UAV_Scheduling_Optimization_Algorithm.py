from GA_based_UAV_Scheduling_Algorithm import GA_based_UAV_Scheduling_Algorithm
from GA_based_Task_Offloading_Algorithm import GA_based_Task_Offloading_Algorithm
from model_compute import total_satisfaction

def Joint_Task_Offloading_and_UAV_Scheduling_Optimization_Algorithm(users, drones, Groups):
    Miter = 25  # 迭代次数
    n = 0
    satisfaction = 0
    no_change_count = 0
    filename = f"GA.txt"
    # 输入为：种群大小L2设置为50.Niter2设置为150.Pc2和pm2分别设置为0.9和0.3.自优化Ns2中爬升数设置为20个,突变操作Nm2中改变基因数设置为10个.
    population_size = 50  # 种群大小L2
    # 定义遗传算法参数
    Niter2 = 350  # 迭代次数Niter2,150
    crossover_rate = 0.9  # 交叉率pc2
    mutation_rate = 0.3  # 变异率pm2
    Ns2 = 20  # 自优化Ns2中爬升数
    Nm2 = 10  # 改变基因数
    sum_time = 3  # 五次没变就停
    assign = 'p'
    scheduling = 'GA'
    for u in users:
        u.alpha = 0.3
    while n < Miter:
        print(f"第{n}次....................................联合优化迭代")
        n += 1
        drones, Groups = GA_based_UAV_Scheduling_Algorithm(drones, Groups, assign, scheduling)
        users, drones, Groups = GA_based_Task_Offloading_Algorithm(users, drones, Groups, population_size, Niter2,
                                                                   crossover_rate,
                                                                   mutation_rate, Ns2, Nm2)
        current_satisfaction = total_satisfaction(Groups)
        if current_satisfaction > satisfaction:
            no_change_count = 0
            satisfaction = current_satisfaction
        else:
            no_change_count += 1
        print(f" 目前最高满意度：{satisfaction}")
        if no_change_count >= sum_time or n == Miter:  # 够次数没变就停,然后返回

            return satisfaction, Groups, drones
