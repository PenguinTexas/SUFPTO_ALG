import numpy as np
import random
from PSO_base import PSOBase
from model_compute import energy_constraint, total_satisfaction

class UAVSchedulingPSO(PSOBase):
    """无人机调度 PSO（离散优化）"""
    def __init__(self, drones, Groups, **kwargs):
        super().__init__(**kwargs)
        self.drones = drones
        self.Groups = Groups
        self.num_groups = len(Groups)
        self.num_drones = len(drones)

    def fitness_function(self, assignment):
        """适应度函数"""
        # 保存原始状态
        original_state = self._save_state()
        # 应用分配
        self._apply_assignment(assignment)
        # 计算满意度
        satisfaction = total_satisfaction(self.Groups)
        # 能量约束惩罚
        penalty = sum(1000 for d in self.drones if len(d.serve_group) > 0 and not energy_constraint(d))
        # 恢复状态
        self._restore_state(original_state)
        return satisfaction - penalty

    def _save_state(self):
        """保存状态"""
        return {
            'drone_serve': {d.k: d.serve_group.copy() for d in self.drones},
            'drone_Q': {d.k: d.Q.copy() for d in self.drones},
            'group_drone': {g.h: g.drone_k for g in self.Groups}
        }

    def _apply_assignment(self, assignment):
        """应用分配方案"""
        for d in self.drones:
            d.serve_group = []
            d.Q = [((0, 0), d.hk)]
        for g in self.Groups:
            g.drone_k = None
        for g_idx, d_idx in enumerate(assignment):
            if 0 <= d_idx < self.num_drones:
                self.drones[d_idx].serve_group.append(self.Groups[g_idx])
                self.Groups[g_idx].drone_k = self.drones[d_idx]
                self.drones[d_idx].Q.append((self.Groups[g_idx].chloc, self.drones[d_idx].hk))

    def _restore_state(self, state):
        """恢复状态"""
        for d in self.drones:
            d.serve_group = state['drone_serve'][d.k]
            d.Q = state['drone_Q'][d.k]
        for g in self.Groups:
            g.drone_k = state['group_drone'][g.h]

    def initialize_particles(self):
        """初始化（离散）"""
        self.particles = []
        for _ in range(self.num_particles):
            position = np.array([random.randint(-1, self.num_drones - 1) for _ in range(self.num_groups)])
            velocity = np.random.uniform(-1, 1, self.num_groups)
            self.particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': float('-inf')
            })

    def update_position(self, particle, iteration):
        """更新位置（离散化）"""
        new_position = particle['position'].astype(float) + particle['velocity']
        new_position = np.round(new_position).astype(int)
        new_position = np.clip(new_position, -1, self.num_drones - 1)
        particle['position'] = new_position

    def run(self):
        """运行优化"""
        self.initialize_particles()
        for p in self.particles:
            fitness = self.fitness_function(p['position'])
            p['best_fitness'] = fitness
            p['best_position'] = p['position'].copy()
        self.global_best['position'] = max(self.particles, key=lambda p: p['best_fitness'])['best_position'].copy()
        self.global_best['fitness'] = max(p['best_fitness'] for p in self.particles)

        for iteration in range(self.num_iterations):
            for p in self.particles:
                self.update_velocity(p, iteration)
                self.update_position(p, iteration)
                fitness = self.fitness_function(p['position'])
                if fitness > p['best_fitness']:
                    p['best_fitness'] = fitness
                    p['best_position'] = p['position'].copy()
                if fitness > self.global_best['fitness']:
                    self.global_best['fitness'] = fitness
                    self.global_best['position'] = p['position'].copy()

        # 应用最优解
        self._apply_assignment(self.global_best['position'])
        # 能量约束修复
        for d in self.drones:
            while len(d.serve_group) > 0 and not energy_constraint(d):
                g = d.serve_group.pop()
                g.drone_k = None
                if len(d.Q) > 1:
                    d.Q.pop()
        return self.drones, self.Groups, self.global_best['fitness']

def PSO_UAV_Scheduling(drones, Groups, **kwargs):
    """快捷调用"""
    pso = UAVSchedulingPSO(drones, Groups, **kwargs)
    return pso.run()