import os
import pickle
from k_means_plus_plus import create_group
from DDPGEnvironment import DDPG_test,DDPG_train
from PSO_based_UAV_Offloading_Scheduling import PSO_optimization

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
                                    # === 检查并训练模型 ===
                                    model_dir = f"model/{d_n}_{u_n}"
                                    actor_path = f"{model_dir}/actor_model.pth"
                                    if not os.path.exists(actor_path):
                                        print(f"训练模型: drones={d_n}, users={u_n}")
                                        # 先训练再测试
                                        DDPG_train(users, drones, Groups)  # 训练并保存模型
                                    # === 加载训练好的模型，评估强化学习算法在测试数据上的性能 ===
                                    satisfaction, Groups, drones = DDPG_test(users, drones, Groups)
                                    drones, satisfaction = PSO_optimization(drones, Groups, users)
                                    print(satisfaction)
                                    # 下面都是写结果部分
                                    desired_part = file_directory.split('/')
                                    content_name = desired_part[1] + f'_{desired_part[2]}'
                                    Argument = desired_part[1].split('_')
                                    content = ([content_name] + ['UTIC'] + Argument +
                                               [f'{satisfaction:.1f}'])
                                    # 使用'a'模式打开文件以进行追加写入
                                    # 直接全部保存到一个文件
                                    with open('result.txt', 'a') as file:
                                        file.write('\t'.join(content) + '\n')


if __name__ == '__main__':
    drone_B = [3]  # 无人机带宽 (MHz), =*1e6Hz
    drone_energy = [70]  # 无人机电量 (Wh), =*3600J
    user_computing_power = [(2, 4)]  # 用户计算能力范围 (MHz), =*1e6Hz
    drone_computing_power = [200]  # 无人机计算能力 (MHz), =*1e6Hz
    drone_velocity = [40]  # 无人机速度 (m/s)
    drone_number = [4]     # 无人机数量
    user_number = [50]     # 用户数量
    instance_test(drone_B, drone_energy, user_computing_power, drone_computing_power, drone_velocity, drone_number,
                  user_number)