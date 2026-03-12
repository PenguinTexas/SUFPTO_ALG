import os
import pickle
import sys
import time



from HA import HA_train, HA_test

from k_means_plus_plus import create_group


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
                                for k in range(5):
                                    file_directory = f"result/{B}_{e}_{u_p}_{d_p}_{v}_{d_n}_{u_n}/{k}"
                                    file_name = f"{B}_{e}_{u_p}_{d_p}_{v}_{d_n}_{u_n}_{k}_users_drones.pkl"
                                    with open(os.path.join(file_directory, file_name), 'rb') as file:
                                        loaded_data = pickle.load(file)
                                        users = loaded_data['users']
                                        drones = loaded_data['drones']
                                    # 算法开始
                                    start = time.time()
                                    Groups = create_group(users, 12, 100)

                                    # satisfaction, Groups, drones = HA_test(users, drones, Groups)
                                    satisfaction, Groups, drones = HA_train(users, drones, Groups)
                                    # file_result = f"{B}_{e}_{u_p}_{d_p}_{v}_{d_n}_{u_n}_{k}_result.pkl"
                                    # # 保存计算的结果以便比较其他目标
                                    # with open(os.path.join(file_directory, file_result), 'wb') as file:
                                    #     data_to_save = {'Groups': Groups, 'drones': drones, 'user': users}
                                    #     pickle.dump(data_to_save, file)
                                    # 下面都是写结果部分
                                    desired_part = file_directory.split('/')
                                    content_name = desired_part[1] + f'_{desired_part[2]}'
                                    Argument = desired_part[1].split('_')
                                    content = [content_name] + ['HA'] + Argument + [
                                        f'{satisfaction:.1f}'] + [f"{(time.time() - start):.0f}"]
                                    # # 使用'a'模式打开文件以进行追加写入
                                    # 直接全部保存到一个文件
                                    with open('result.txt', 'a') as file:
                                        file.write('\t'.join(content) + '\n')


if __name__ == '__main__':
    # 729=3*3*3*3*3*3
    drone_B = [1]  # MHz,=*1e6Hz
    drone_energy = [60, 70, 80]  # Wh,=*3600J
    user_computing_power = [(2, 4), (4, 6), (6, 8)]  # MHz,=*1e6Hz
    drone_computing_power = [100, 150, 200]  # MHz,=*1e6Hz
    drone_velocity = [40, 50, 60]  # m/s
    drone_number = [4,8,12]
    user_number = [100]
    instance_test(drone_B, drone_energy, user_computing_power, drone_computing_power, drone_velocity, drone_number,
                  user_number)
