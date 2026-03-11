import math
import random

class User:
    def __init__(self, j, xj, yj, Emaxj, Tj, beta_j, Dj, Vj, T, alpha, f_j, phi_t, phi_e):
        self.j = j  # 第j个用户
        self.xj = xj  # 用户位置 x 坐标
        self.yj = yj  # 用户位置 y 坐标
        self.Wj = (self.xj, self.yj)  # 用户坐标
        self.Emaxj = Emaxj  # 最大能量(Wh)
        self.Erj = random.uniform(0, Emaxj)  # 剩余能量(Wh)
        self.Tj = Tj  # 理想容忍延迟，即用户期望的最佳任务完成时间
        self.beta_j = beta_j  # 用户容忍延迟常数因子
        self.Dj = Dj  # 任务大小(bits)
        self.Vj = Vj  # 任务结果大小(bits)
        self.T = T  # 服务时长,period,duration
        self.alpha = alpha  # 任务卸载到无人机的比例
        self.f_j = f_j  # 用户的计算能力(Hz)
        k_j = 1e-20  # 正系数
        self.P_comp_j = k_j * (self.f_j ** 3)  # 用户的计算功率

        if not (0 < phi_t < 1 and 0 < phi_e < 1 and phi_t + phi_e == 1):
            self.valid = False
            self.error_message = "权值无效"
        else:
            self.valid = True
            self.phi_t = phi_t
            self.phi_e = phi_e
            # 计算任务优先级并设置成员特征集
            self.delta_j = self.calculate_priority()  #
            self.zj = (self.delta_j, (xj, yj))  # 成员特征集 (δj, wj)

    def Task(self):  # 任务需求phi_t,phi_e别是任务最大延迟和用户剩余能量的权值,∈(0，1)
        return True

    def calculate_priority(self):
        delta_j = self.phi_e * math.exp(-self.Erj / self.Emaxj) + self.phi_t * math.exp(-self.beta_j * self.Tj / self.T)
        return round(delta_j, 5)


class Group:
    def __init__(self, h, users):
        self.h = h  # 第h组
        self.users = users  # 组内的用户列表
        # 所属组群特征集 (chpri, chloc)
        self.ch = self.calculate_features()
        self.chpri = self.ch[0]  # 组优先级
        self.chloc = self.ch[1]  # 组的中心坐标
        self.drone_k = None  # 所属无人机

    def calculate_features(self):
        chpri = (1 / len(self.users)) * sum(float(i.zj[0]) for i in self.users)
        chloc_x = (1 / len(self.users)) * sum(i.zj[1][0] for i in self.users)
        chloc_y = (1 / len(self.users)) * sum(i.zj[1][1] for i in self.users)
        ch = (round(chpri, 5), (round(chloc_x, 5), round(chloc_y, 5)))
        return ch

class Drone:
    def __init__(self, k, hk, Vk, f_k, Emaxk, Bk):
        self.k = k
        self.hk = hk  # 无人机高度
        self.Vk = Vk  # 无人机速度
        self.f_k = f_k  # 无人机的计算能力(MHz)
        self.Emaxk = Emaxk  # 最大能量(Wh)
        s_k = 1e-20  # 表示无人机k的有效电容系数
        self.P_comp_k = s_k * (self.f_k ** 3)  # 无人机的计算功率
        # 服务的用户各组
        self.serve_group = []  # c_k
        self.Q = [((0, 0), self.hk)]  # 无人机的轨迹
        self.Bk = Bk  # 可用总带宽

