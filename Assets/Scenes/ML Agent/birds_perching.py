import numpy as np
import matplotlib.pyplot as plt

# Constants and initial conditions
k = 1.6
kd_alpha = 0.4
initial_velocity = 4.5

# Initial and target positions
p0 = np.array([0, 0, 10])
ptd = np.array([3, 4, 0])

# Time settings
td = 5  # total time duration
t = np.linspace(0, td, 500)

# Compute distance gap (d(t)) based on equation (13)
tau_0 = (np.linalg.norm(ptd - p0)) / initial_velocity
d_t = initial_velocity * tau_0 * (1 - t / td)**(1/k)

# Compute angular gap alpha(t) based on equation (15)
alpha_0 = np.pi / 4  # initial angle gap, 45 degrees as example
alpha_t = alpha_0 * (d_t / d_t[0])**(1/kd_alpha)

# Compute position p(t) based on equation (16)
p_t = np.zeros((len(t), 3))
for i in range(len(t)):
    M1 = (np.sin(alpha_t[0]) - np.sin(alpha_t[i])) / np.sin(alpha_t[0])
    M2 = np.sin(alpha_t[i]) / np.sin(alpha_t[0])
    M3 = np.array([0, 0, d_t[i] * np.sin(alpha_t[i])])
    p_t[i] = M1 * ptd + M2 * p0 + M3

# Plotting the trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(p_t[:, 0], p_t[:, 1], p_t[:, 2], label='Perching Trajectory (Case 2)')
ax.scatter(*p0, color='red', label='Initial Position')
ax.scatter(*ptd, color='green', label='Target Position')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('UAV Perching Trajectory with Pitch Angle Coupling')
plt.show()