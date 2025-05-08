import numpy as np
import matplotlib.pyplot as plt

# === Constants and Initial Conditions ===
k = 0.6                      # distance‐decay shape parameter
kd_alpha = 0.4               # angle‐decay shape parameter
initial_velocity = 4.5       # m/s

# Initial (p0) and target/perch (ptd) positions
p0  = np.array([0.0, 0.0, 0.0])
ptd = np.array([13.0, 14.0, 0.0])

# === Phase 1: Perching Maneuver ===
td = 5.0                     # total perch duration (s)
t1 = np.linspace(0, td, 500)

tau_0 = np.linalg.norm(ptd - p0) / initial_velocity
d_t = initial_velocity * tau_0 * (1 - t1/td)**(1/k)
alpha_0 = np.pi / 4
alpha_t = alpha_0 * (d_t / d_t[0])**(1/kd_alpha)

p1 = np.zeros((len(t1), 3))
for i, ti in enumerate(t1):
    M1 = (np.sin(alpha_t[0]) - np.sin(alpha_t[i])) / np.sin(alpha_t[0])
    M2 = np.sin(alpha_t[i]) / np.sin(alpha_t[0])
    M3 = np.array([0.0, 0.0, d_t[i] * np.sin(alpha_t[i])])
    p1[i] = M1*ptd + M2*p0 + M3

# === Phase 2a: 4 m Horizontal Forward Approach ===
# Compute horizontal direction unit vector
h = ptd - p0
h[2] = 0
h_unit = h / np.linalg.norm(h)

dist_approach = 4.0                  # meters to fly forward at perch height
t2a_duration = dist_approach / initial_velocity
n2a = 100                            # sample points for approach
t2a = np.linspace(td, td + t2a_duration, n2a)

# Build the forward approach trajectory (constant Z = ptd[2])
p2a = np.array([
    ptd + h_unit * initial_velocity * (ti - td)
    for ti in t2a
])

# === Phase 2b: Fly‐Away (Climb‐Out) Maneuver ===
t_exit = 5.0                         # duration of climb‐out (s)
n2b = 200                            # sample points for climb‐out
t2b = np.linspace(td + t2a_duration, td + t2a_duration + t_exit, n2b)

exit_angle_deg = 30.0
φ_exit = np.deg2rad(exit_angle_deg)
v_exit = initial_velocity

u_exit = np.array([
    np.cos(φ_exit) * h_unit[0],
    np.cos(φ_exit) * h_unit[1],
    np.sin(φ_exit)
])

# New “perch‐plus‐4m” point is the start of the climb segment
ptd2 = ptd + h_unit * dist_approach

p2b = np.array([
    ptd2 + v_exit * u_exit * (ti - (td + t2a_duration))
    for ti in t2b
])

# === Concatenate Full Trajectory ===
t_full = np.concatenate([t1, t2a, t2b])
p_full = np.vstack([p1, p2a, p2b])

# === Plotting ===
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')

ax.plot(p1[:,0], p1[:,1], p1[:,2], label='Perch Trajectory', linewidth=2)
ax.plot(p2a[:,0], p2a[:,1], p2a[:,2], label='Forward Approach', linewidth=2)
ax.plot(p2b[:,0], p2b[:,1], p2b[:,2], label='Fly‐Away', linestyle='--', linewidth=2)

ax.scatter(*p1[0],  color='red',   label='Start (p0)',   s=50)
ax.scatter(*ptd, color='green', label='Perch Point',   s=50)
ax.scatter(*ptd2, color='orange', label='Post‐Approach Point', s=50)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('UAV Touch-Down, Forward Approach & Fly-Away')
ax.legend()
plt.tight_layout()
plt.show()
