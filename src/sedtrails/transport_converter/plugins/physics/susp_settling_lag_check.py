# Particle state update example for SedTRAILS: suspended vs bedload with probabilistic transitions
import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, state="bedload"):
        self.position = position  # 2D position (x, y) on the bed
        self.state = state  # "suspended" or "bedload"

class SedimentModel:
    def __init__(self, particles, shear_stress_func, dt, tau_c, tau_d, lambda_s_max, lambda_e_max, w_s, depth):
        self.particles = particles
        self.tau = shear_stress_func  # function: position -> tau
        self.dt = dt  # timestep [s]
        self.tau_c = tau_c  # critical entrainment threshold
        self.tau_d = tau_d  # low-shear threshold for deposition
        self.lambda_s_max = lambda_s_max  # max settling rate (suspended -> bedload)
        self.lambda_e_max = lambda_e_max  # max entrainment rate (bedload -> suspended)
        self.w_s = w_s  # settling velocity [m/s]
        self.depth = depth  # water column depth [m]
        self.state_history = []

    def update_particle_states(self):
        suspended_count = 0
        for p in self.particles:
            tau_local = self.tau(p.position)
            r = np.random.rand()

            if p.state == "bedload":
                if tau_local > self.tau_c:
                    lambda_e = self.lambda_e_max * (tau_local / self.tau_c - 1)
                    P_entrain = 1 - np.exp(-max(lambda_e, 0) * self.dt)
                    if r < P_entrain:
                        p.state = "suspended"

            elif p.state == "suspended":
                if tau_local < self.tau_c:
                    if tau_local < self.tau_d:
                        lambda_s = self.lambda_s_max
                    else:
                        lambda_s = self.lambda_s_max * (self.tau_c - tau_local) / (self.tau_c - self.tau_d)
                    P_settle = 1 - np.exp(-lambda_s * self.dt)
                    if r < P_settle:
                        p.state = "bedload"

            if p.state == "suspended":
                suspended_count += 1

        self.state_history.append(suspended_count / len(self.particles))

    def advect_particles(self, flow_velocity_func):
        for p in self.particles:
            u = flow_velocity_func(p.position)
            if p.state == "suspended":
                p.position += u * self.dt
            elif p.state == "bedload":
                p.position += 0.3 * u * self.dt

    def plot_particle_distribution(self):
        x_bedload = [p.position[0] for p in self.particles if p.state == "bedload"]
        x_suspended = [p.position[0] for p in self.particles if p.state == "suspended"]
        y_bedload = [p.position[1] for p in self.particles if p.state == "bedload"]
        y_suspended = [p.position[1] for p in self.particles if p.state == "suspended"]

        plt.figure(figsize=(10, 4))
        plt.scatter(x_bedload, y_bedload, c='brown', label='Bedload', alpha=0.6)
        plt.scatter(x_suspended, y_suspended, c='blue', label='Suspended', alpha=0.6)
        plt.xlabel("x position (m)")
        plt.ylabel("y position (m)")
        plt.title("Particle Positions by State")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_state_history(self):
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(len(self.state_history)) * self.dt / 60, self.state_history)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Fraction Suspended")
        plt.title("Suspended Particle Fraction Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example shear stress and flow velocity functions
def example_shear_stress(position):
    x, y = position
    return 0.07 + 0.03 * np.sin(x / 10)

def example_flow_velocity(position):
    return np.array([0.5, 0.0])

# Example setup
particles = [Particle(np.array([x, 0.0])) for x in np.linspace(0, 100, 200)]
model = SedimentModel(
    particles,
    shear_stress_func=example_shear_stress,
    dt=60,
    tau_c=0.08,
    tau_d=0.04,
    lambda_s_max=0.001,
    lambda_e_max=0.005,
    w_s=1e-3,
    depth=1.0
)

# Time loop
for step in range(100):
    model.update_particle_states()
    model.advect_particles(example_flow_velocity)

# Diagnostics
model.plot_state_history()
model.plot_particle_distribution()
