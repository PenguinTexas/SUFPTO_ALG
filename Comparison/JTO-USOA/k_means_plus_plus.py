import random

import numpy as np


from class_member import Group
import sys

def k_means_plus_plus(users, k, max_iterations):
    # 步骤 1：随机选择一个用户作为初始中心
    temp = random.choice(users)
    centers_feature = [(temp.delta_j, (temp.xj, temp.yj))]

    # 步骤 2：使用 K-means++ 初始化选择其余的 k-1 个中心
    for _ in range(k - 1):
        distances = []
        for user in users:
            # 计算每个用户到最近中心的距离的平方
            min_distance = min(
                [np.linalg.norm(np.array(user.Wj) - np.array(center[1])) ** 2 for center in centers_feature])
            distances.append(min_distance)
        # 根据距离的平方选择下一个中心
        probabilities = [d / sum(distances) for d in distances]
        new_center = random.choices(users, probabilities)[0]
        centers_feature.append((new_center.delta_j, (new_center.xj, new_center.yj)))

    # 步骤 3：将用户分配给最近的中心并更新群组
    groups = [[] for _ in range(k)]
    for user in users:
        min_distance = float('inf')
        assigned_group = None
        for i, center in enumerate(centers_feature):
            distance = calculate_distance(user, center)
            if distance < min_distance:
                min_distance = distance
                assigned_group = i
        groups[assigned_group].append(user)

    # 步骤 4：更新中心并重复步骤3 直到收敛
    for _ in range(max_iterations):

        # new_centers = []
        new_centers_feature = []
        for group in groups:
            if group:
                new_center_feature = calculate_mean_feature(group)
                new_centers_feature.append(new_center_feature)
            else:
                # 空，重新选一个
                print("a")
                temp = random.choice(users)
                new_centers_feature.append((temp.delta_j, (temp.xj, temp.yj)))

        if new_centers_feature == centers_feature:
            break
        centers_feature = new_centers_feature

        groups = [[] for _ in range(k)]
        # 重新分配
        for user in users:
            min_distance = float('inf')
            assigned_group = None
            for i, center in enumerate(centers_feature):
                distance = calculate_distance(user, center)
                if distance < min_distance:
                    min_distance = distance
                    assigned_group = i
            groups[assigned_group].append(user)

    return groups, centers_feature


def calculate_distance(user1, center):
    return ((user1.xj - center[1][0]) ** 2 + (user1.yj - center[1][1]) ** 2) ** 0.5


def calculate_mean_feature(group):
    total_x = sum(user.xj for user in group)
    total_y = sum(user.yj for user in group)
    total_priority = sum(user.delta_j for user in group)
    mean_x = round(total_x / len(group), 5)
    mean_y = round(total_y / len(group), 5)
    mean_priority = total_priority / len(group)
    mean_feature = (round(mean_priority, 5), (mean_x, mean_y))
    return mean_feature


def create_group(users, k, max_iterations):  # 分组数，迭代次数
    Groups = []  # 组对象列表,文中的C
    # 执行 K-means++ 分组算法
    if k > len(users):
        print("分组数大于用户数,请重新输入分组数")
        sys.exit()
    groups, centers_features = k_means_plus_plus(users, k, max_iterations)  # 得到分组结果和每个组的中心坐标
    # 在每个组内对用户列表按照优先级进行排序
    for u in groups:
        u.sort(key=lambda user: user.delta_j, reverse=True)
    for i, g in enumerate(groups):  # 创建组对象
        group = Group(i, g)
        Groups.append(group)
    # 将groups每个组按照优先级重新排序
    Groups.sort(key=lambda group: group.chpri, reverse=True)
    Groups = np.array(Groups)
    return Groups

