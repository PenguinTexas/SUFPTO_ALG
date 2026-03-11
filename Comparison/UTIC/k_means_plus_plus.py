import sys
import random
from class_member import Group
from model_compute import calculate_distance

# mean_feature = (组理想容忍度，(组中心坐标))
def calculate_mean_feature(group):
    total_x = sum(user.xj for user in group)
    total_y = sum(user.yj for user in group)
    total_Tj = sum(user.Tj for user in group)
    mean_x = round(total_x / len(group), 5)
    mean_y = round(total_y / len(group), 5)
    mean_Tj = total_Tj / len(group)
    mean_feature = (round(mean_Tj, 5), (mean_x, mean_y))
    return mean_feature

# K-Means分组
def k_means_plus_plus(users, k, max_iterations):
    random.seed(0)
    # 步骤 1：随机选择一个用户作为初始中心
    temp = random.choice(users)
    centers_feature = [(temp.Tj, (temp.xj, temp.yj))]
    # 步骤 2：使用 K-means++ 初始化选择其余的 k-1 个中心
    for _ in range(k - 1):
        distances = []
        for user in users:
            # 计算每个用户到最近中心的距离的平方
            min_distance = min([calculate_distance(user.Wj, center[1]) ** 2 for center in centers_feature])
            distances.append(min_distance)
        # 根据距离的平方选择下一个中心
        probabilities = [d / sum(distances) for d in distances]
        new_center = random.choices(users, probabilities)[0]
        centers_feature.append((new_center.Tj, (new_center.xj, new_center.yj)))
    # 步骤 3：将用户分配给最近的中心并更新群组
    groups = [[] for _ in range(k)]
    for user in users:
        min_distance = float('inf')
        assigned_group = None
        for i, center in enumerate(centers_feature):
            distance = calculate_distance(user.Wj, center[1])
            if distance < min_distance:
                min_distance = distance
                assigned_group = i
        groups[assigned_group].append(user)
    # 步骤 4：更新中心并重复步骤3 直到收敛
    for _ in range(max_iterations):
        new_centers_feature = []
        for group in groups:
            if group:
                new_center_feature = calculate_mean_feature(group)
                new_centers_feature.append(new_center_feature)
            else:
                # 空，重新选一个
                temp = random.choice(users)
                new_centers_feature.append((temp.Tj, (temp.xj, temp.yj)))
        if new_centers_feature == centers_feature:
            break
        centers_feature = new_centers_feature
        groups = [[] for _ in range(k)]
        # 重新分配
        for user in users:
            min_distance = float('inf')
            assigned_group = None
            for i, center in enumerate(centers_feature):
                distance = calculate_distance(user.Wj, center[1])
                if distance < min_distance:
                    min_distance = distance
                    assigned_group = i
            groups[assigned_group].append(user)
    return groups, centers_feature

# 创建组函数
def create_group(users, k, max_iterations):  # 分组数，迭代次数
    Groups = []  # 组对象列表,文中的C
    if k > len(users):
        print("分组数大于用户数")
        sys.exit()
    # 执行 K-means++ 分组算法，得到分组结果和每个组的中心坐标
    groups, centers_features = k_means_plus_plus(users, k, max_iterations)
    # 在每个组内对用户列表按照最大容忍延迟进行升序排序（紧迫任务优先）
    for u in groups:
        u.sort(key=lambda user: user.Tj * user.beta_j)
    # 创建组对象
    for i, g in enumerate(groups):
        group = Group(i, g)
        Groups.append(group)
    return Groups
