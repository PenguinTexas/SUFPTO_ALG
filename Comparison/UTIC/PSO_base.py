import numpy as np

class PSOBase:
    """通用 PSO 基类"""
    def __init__(self, num_particles=50, num_iterations=100,
                 w_max=0.9, w_min=0.4, c1=1.5, c2=1.5,
                 v_max=0.1, v_min=-0.1):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.v_min = v_min
        self.particles = []
        self.global_best = {'position': None, 'fitness': float('-inf')}

    def initialize_particles(self, num_dimensions, pos_bounds=(0, 1)):
        """初始化粒子群"""
        self.particles = []
        for _ in range(self.num_particles):
            position = np.random.uniform(pos_bounds[0], pos_bounds[1], num_dimensions)
            velocity = np.random.uniform(self.v_min, self.v_max, num_dimensions)
            self.particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': float('-inf')
            })

    def update_velocity(self, particle, iteration):
        """更新粒子速度"""
        w = self.w_max - (self.w_max - self.w_min) * (iteration / self.num_iterations)
        r1, r2 = np.random.rand(len(particle['position'])), np.random.rand(len(particle['position']))
        particle['velocity'] = (
                w * particle['velocity'] +
                self.c1 * r1 * (particle['best_position'] - particle['position']) +
                self.c2 * r2 * (self.global_best['position'] - particle['position'])
        )
        particle['velocity'] = np.clip(particle['velocity'], self.v_min, self.v_max)

    def update_position(self, particle, iteration, pos_bounds=(0, 1)):
        """更新粒子位置（边界反弹）"""
        particle['position'] += particle['velocity']
        for d in range(len(particle['position'])):
            if particle['position'][d] < pos_bounds[0]:
                particle['position'][d] = pos_bounds[0]
                particle['velocity'][d] = -particle['velocity'][d] * (1 - iteration / self.num_iterations)
            elif particle['position'][d] > pos_bounds[1]:
                particle['position'][d] = pos_bounds[1]
                particle['velocity'][d] = -particle['velocity'][d] * (1 - iteration / self.num_iterations)

    def evaluate_and_update(self, particle, fitness_func):
        """评估适应度并更新最优"""
        fitness = fitness_func(particle['position'])
        if fitness > particle['best_fitness']:
            particle['best_fitness'] = fitness
            particle['best_position'] = particle['position'].copy()
        if fitness > self.global_best['fitness']:
            self.global_best['fitness'] = fitness
            self.global_best['position'] = particle['position'].copy()
        return fitness