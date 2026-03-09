import os
import pickle
from k_means_plus_plus import create_group
from offloading_scheduling import SA_path_offloading, PSO_single_optimization
from model_compute import T_total,E_save, E_msave

def analyze_user_task_completion(users, Groups, lambda_1):
    """
    分析用户任务完成时间情况并输出到文件
    """
    # 构建用户到组的映射
    user_to_group = {}
    for group in Groups:
        for user in group.users:
            user_to_group[user] = group

    # 初始化统计变量
    total_users = len(users)
    condition1_count = 0  # T_total_value <= beta_j * Tj 且 T_diff <= 0
    condition2_count = 0  # T_total_value <= beta_j * Tj 且 T_diff > 0
    condition3_count = 0  # T_total_value > beta_j * Tj

    condition2_ratios = []  # 存储T_diff / (beta_j * Tj - Tj)的值

    # 遍历所有用户
    for user in users:
        group = user_to_group.get(user)
        if not group:
            continue

        # 找到用户在组中的索引
        try:
            user_index = group.users.index(user)
        except ValueError:
            continue

        # 获取相关参数
        beta_j = user.beta_j
        Tj = user.Tj
        T_total_value = T_total(user_index, group)

        # 处理无穷大的情况
        if T_total_value == float('inf'):
            condition3_count += 1
            continue

        T_diff = T_total_value - Tj

        # 分类统计
        if T_total_value <= beta_j * Tj:
            if T_diff <= 0:
                condition1_count += 1
            else:
                condition2_count += 1
                denominator = beta_j * Tj - Tj
                if denominator > 0:  # 避免除零错误
                    ratio = T_diff / denominator
                    condition2_ratios.append(ratio)
        else:
            condition3_count += 1

    # 计算平均值
    avg_ratio = sum(condition2_ratios) / len(condition2_ratios) if condition2_ratios else 0

    # 计算满意度
    S_a = (lambda_1 - lambda_1 ** avg_ratio)/(lambda_1 - 1)
    total_S_a = (condition1_count + condition2_count * S_a) / total_users

    return {
        'total_users': total_users,
        'condition1_count': condition1_count,
        'condition2_count': condition2_count,
        'condition3_count': condition3_count,
        'avg_ratio': avg_ratio,
        'total_S_a': total_S_a
    }

def analyze_user_energy_alpha(users, Groups, lambda_2, gamma):
    """
    分析用户能量状态和卸载比例情况并输出到文件
    """
    # 构建用户到组的映射
    user_to_group = {}
    for group in Groups:
        for user in group.users:
            user_to_group[user] = group

    # 初始化统计变量
    total_users = len(users)
    condition1_count = 0  # 0 < Erj < gamma * Emaxj 且 alpha == 0
    condition2_count = 0  # 0 < Erj < gamma * Emaxj 且 alpha != 0
    condition3_count = 0  # Erj >= gamma * Emaxj
    condition4_count = 0  # Erj <= 0

    condition2_values = []  # 存储(1 - (E_save(j, Group) / E_msave(j, Group)))的值

    # 遍历所有用户
    for user in users:
        group = user_to_group.get(user)
        if not group:
            continue

        # 找到用户在组中的索引
        try:
            user_index = group.users.index(user)
        except ValueError:
            continue

        # 获取相关参数
        Erj = user.Erj
        Emaxj = user.Emaxj
        alpha = user.alpha

        # 分类统计
        if 0 < Erj < gamma * Emaxj:
            if alpha == 0:
                condition1_count += 1
            else:
                condition2_count += 1
                # 计算(1 - (E_save(j, Group) / E_msave(j, Group)))
                E_save_val = E_save(user_index, group)
                E_msave_val = E_msave(user_index, group)

                if E_msave_val > 0:  # 避免除零错误
                    ratio = E_save_val / E_msave_val
                    # 确保比率在合理范围内
                    ratio = max(0, min(1, ratio))
                    value = 1 - ratio
                    condition2_values.append(value)
        elif Erj >= gamma * Emaxj:
            condition3_count += 1
        else:  # Erj <= 0
            condition4_count += 1

    # 计算平均值
    avg_value = sum(condition2_values) / len(condition2_values) if condition2_values else 0

    # 计算满意度
    S_a = (lambda_2 - lambda_2 ** avg_value) / (lambda_2 - 1)
    total_S_e = (condition3_count + condition2_count * S_a) / total_users

    return {
        'total_users': total_users,
        'condition1_count': condition1_count,
        'condition2_count': condition2_count,
        'condition3_count': condition3_count,
        'condition4_count': condition4_count,
        'avg_value': avg_value,
        'total_S_e': total_S_e
    }

def instance_test(drone_b, drone_Energy, user_Computing_power, drone_Computing_power, drone_Velocity, drone_Number,
                  user_Number):
    # 命名规则：无人机带宽_无人机电量_用户计算能力_无人机计算能力_无人机速度_无人机数量_用户数量
    for B in drone_b:
        for e in drone_Energy:
            for u_p in user_Computing_power:
                for d_p in drone_Computing_power:
                    for v in drone_Velocity:
                        for d_n in drone_Number:
                            for u_n in user_Number:
                                for k in range(1):  # 每个组合运行 1 次
                                    # 加载测试数据
                                    file_directory = f"instance/{B}_{e}_{u_p}_{d_p}_{v}_{d_n}_{u_n}/{k}"
                                    file_name = f"{B}_{e}_{u_p}_{d_p}_{v}_{d_n}_{u_n}_{k}_users_drones.pkl"
                                    with open(os.path.join(file_directory, file_name), 'rb') as file:
                                        data = pickle.load(file)
                                        users, drones = data['users'], data['drones']
                                    # 算法开始
                                    a = 1.5  # 分组数
                                    groups_k = round(a * d_n)
                                    Groups = create_group(users, groups_k, 100)
                                    satisfaction, Groups, drones, false_time = SA_path_offloading(Groups, drones)
                                    drones, satisfaction = PSO_single_optimization(drones, Groups, users)
                                    # 分析用户任务完成时间
                                    t = analyze_user_task_completion(users, Groups, 2)
                                    # 分析用户能量状态和卸载比例
                                    e = analyze_user_energy_alpha(users, Groups, 2, 0.8)
                                    print(f"用户任务完成时间分析:")
                                    print(
                                        f"  T_total_value <= beta_j * Tj 且 T_diff <= 0: {t['condition1_count']} 用户")
                                    print(
                                        f"  T_total_value <= beta_j * Tj 且 T_diff > 0: {t['condition2_count']} 用户")
                                    print(
                                        f"  T_diff > 0 用户的平均 T_diff / (beta_j * Tj - Tj): {t['avg_ratio']:.4f}")
                                    print(
                                        f"  T_total_value > beta_j * Tj: {t['condition3_count']} 用户")
                                    print(f"  S_a: {t['total_S_a']} 用户")
                                    print(f"用户能量状态和卸载比例分析:")
                                    print(
                                        f"  0 < Erj < gamma * Emaxj 且 alpha == 0: {e['condition1_count']} 用户")
                                    print(
                                        f"  0 < Erj < gamma * Emaxj 且 alpha != 0: {e['condition2_count']} 用户")
                                    print(
                                        f"  alpha != 0 用户的平均 (1 - (E_save / E_msave)): {e['avg_value']:.4f}")
                                    print(f"  Erj >= gamma * Emaxj: {e['condition3_count']} 用户")
                                    print(f"  Erj <= 0: {e['condition4_count']} 用户")
                                    print(f"  S_e: {e['total_S_e']} 用户")
"""
实验对比
带宽 1,2,3            UAV最大电量 60，70，80            用户计算能力，[2,4][4,6][6,8]
无人机计算能力：100，150，200         无人机飞行速度：40，50，60            无人机数量：4，8，12            用户数量：50，100，150，200
"""
if __name__ == '__main__':
    drone_B = [1, 2, 3]  # 无人机带宽 (MHz), =*1e6Hz
    drone_energy = [60, 70, 80]  # 无人机电量 (Wh), =*3600J
    user_computing_power = [(2, 4), (4, 6), (6, 8)]  # 用户计算能力范围 (MHz), =*1e6Hz
    drone_computing_power = [100, 150, 200]  # 无人机计算能力 (MHz), =*1e6Hz
    drone_velocity = [40, 50, 60]  # 无人机速度 (m/s)
    drone_number = [4, 8, 12]     # 无人机数量
    user_number = [50, 100, 150, 200]     # 用户数量
    instance_test(drone_B, drone_energy, user_computing_power, drone_computing_power, drone_velocity, drone_number,
                  user_number)