import math
import random
import time
import numpy as np
from PSO_base import PSOBase
from model_compute import calculate_user_satisfaction, total_satisfaction, energy_constraint

class OffloadingPSO(PSOBase):
    """任务卸载 PSO（连续优化）"""
    def __init__(self, group, **kwargs):
        super().__init__(**kwargs)
        self.group = group
        self.num_users = len(group.users)

    def fitness_function(self, solution):
        """适应度函数"""
        for u, s in zip(self.group.users, solution):
            u.alpha = s
        if not energy_constraint(self.group.drone_k):
            return 0
        return sum(calculate_user_satisfaction(j, self.group) for j in range(self.num_users)) / self.num_users

    def initialize_particles(self, initial_solution=None):
        """初始化粒子群"""
        self.particles = []
        for _ in range(self.num_particles):
            if initial_solution is not None:
                position = initial_solution.copy()
            else:
                position = np.random.uniform(0, 1, self.num_users)
            velocity = np.random.uniform(self.v_min, self.v_max, self.num_users)
            self.particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': self.fitness_function(position)
            })

    def update_position(self, particle, iteration):
        """更新位置（边界反弹）"""
        particle['position'] += particle['velocity']
        for d in range(len(particle['position'])):
            if particle['position'][d] < 0:
                particle['position'][d] = 0
                particle['velocity'][d] = -particle['velocity'][d] * (1 - iteration / self.num_iterations)
            elif particle['position'][d] > 1:
                particle['position'][d] = 1
                particle['velocity'][d] = -particle['velocity'][d] * (1 - iteration / self.num_iterations)

    def run(self, initial_solution=None):
        """运行优化"""
        self.initialize_particles(initial_solution)
        for p in self.particles:
            self.evaluate_and_update(p, self.fitness_function)
        self.global_best['position'] = max(self.particles, key=lambda p: p['best_fitness'])['best_position'].copy()
        self.global_best['fitness'] = max(p['best_fitness'] for p in self.particles)

        for iteration in range(self.num_iterations):
            for p in self.particles:
                self.update_velocity(p, iteration)
                self.update_position(p, iteration)
                self.evaluate_and_update(p, self.fitness_function)

        return self.global_best['position'], self.global_best['fitness']

def offloading_modify_method(group, solution):
    """修正卸载比例满足能量约束"""
    solution = solution.copy()
    for _ in range(100):
        max_idx = np.argmax(solution)
        reduction = random.uniform(0, solution[max_idx])
        solution[max_idx] = max(solution[max_idx] - reduction, 0)
        for u, s in zip(group.users, solution):
            u.alpha = s
        if energy_constraint(group.drone_k):
            return solution
    # 全零解
    for u in group.users:
        u.alpha = 0
    return np.zeros(len(group.users)) if energy_constraint(group.drone_k) else False

def standard_pso(group, initial_solution=None, **kwargs):
    """快捷调用"""
    pso = OffloadingPSO(group, **kwargs)
    return pso.run(initial_solution)

def PSO_optimization(drones, Groups, users):
    """最后优化卸载决策"""
    for d in drones:
        for g in d.serve_group:
            current_solution = np.array([u.alpha for u in g.users])
            solution, fitness = standard_pso(g, initial_solution=current_solution)
            for u, s in zip(g.users, solution):
                u.alpha = s
    # 移除满意度为 0 的组
    for g in Groups:
        if g.drone_k is not None:
            g_satisfaction = sum(calculate_user_satisfaction(j, g) for j in range(len(g.users)))
            if g_satisfaction == 0:
                g.drone_k.serve_group = [group for group in g.drone_k.serve_group if group != g]
                g.drone_k = None
    return drones, total_satisfaction(Groups)