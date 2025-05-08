import numpy as np
import matplotlib.pyplot as plt

# === Constants and Initial Conditions ===
k = 0.6                      # distance‐decay shape parameter
kd_alpha = 0.4               # angle‐decay shape parameter
initial_velocity = 4.5       # m/s

# Initial (p0) and target/perch (ptd) positions
p0  = np.array([0.0, 0.0, 0.0])
ptd = np.array([13.0, 14.0,  0.0])

# === Phase 1: Perching Maneuver ===
td = 5.0                     # total perch duration (s)
t1 = np.linspace(0, td, 500)

# characteristic time to close straight‐line gap
tau_0 = np.linalg.norm(ptd - p0) / initial_velocity

# distance gap d(t)
d_t = initial_velocity * tau_0 * (1 - t1/td)**(1/k)

# angle gap α(t), assuming initial α₀ = 45°
alpha_0 = np.pi / 4
alpha_t = alpha_0 * (d_t / d_t[0])**(1/kd_alpha)

# build perching trajectory p1(t)
p1 = np.zeros((len(t1), 3))
for i, ti in enumerate(t1):
    M1 = (np.sin(alpha_t[0]) - np.sin(alpha_t[i])) / np.sin(alpha_t[0])
    M2 = np.sin(alpha_t[i]) / np.sin(alpha_t[0])
    M3 = np.array([0.0, 0.0, d_t[i] * np.sin(alpha_t[i])])
    p1[i] = M1*ptd + M2*p0 + M3

# === Phase 2: Fly‐Away Maneuver ===
t_exit = 5.0                # duration of fly‐away (s)
n2     = 200                # sample points for departure
t2     = np.linspace(td, td + t_exit, n2)

# horizontal approach direction
h = ptd - p0
h[2] = 0
h_unit = h / np.linalg.norm(h)

# exit flight‐path angle & speed
exit_angle_deg = 30.0
φ_exit = np.deg2rad(exit_angle_deg)
v_exit = initial_velocity     # reuse same speed

# 3D unit vector for climb‐out
u_exit = np.array([
    np.cos(φ_exit) * h_unit[0],
    np.cos(φ_exit) * h_unit[1],
    np.sin(φ_exit)
])

# build fly‐away trajectory p2(t)
p2 = np.array([
    ptd + v_exit * u_exit * (ti - td)
    for ti in t2
])

# === Concatenate Full Trajectory ===
t_full = np.concatenate([t1, t2])
p_full = np.vstack([p1, p2])

# === Plotting ===
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')

# Perch and fly-away segments
ax.plot(p1[:,0], p1[:,1], p1[:,2], label='Perch Trajectory', linewidth=2)
ax.plot(p2[:,0], p2[:,1], p2[:,2], label='Fly-Away Trajectory', linestyle='--', linewidth=2)

# Markers for p0 and ptd
# ax.scatter(*p0,  color='red',   label='Start (p0)',   s=50)
ax.scatter(*ptd, color='green', label='Perch Point',   s=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('UAV Touch-Down & Fly-Away Trajectory')
ax.legend()
plt.tight_layout()
plt.show()
