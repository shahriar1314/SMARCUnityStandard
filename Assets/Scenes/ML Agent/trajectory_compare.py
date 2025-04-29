import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d

# Load dataset with episodes
def load_all_episodes(filename):
    all_episodes = []
    episode = []
    with open(filename, 'r') as f:
        for line in f:
            if "New Episode" in line:
                if episode:
                    all_episodes.append(np.array(episode))
                    episode = []
            elif line.strip():
                try:
                    ep_id, x, y, z = map(float, line.strip().split(','))
                    episode.append([x - 1295, y + 3, z - 1155])
                except ValueError:
                    continue
        if episode:
            all_episodes.append(np.array(episode))
    return all_episodes

# Load single trajectory file and split it into episodes if it has "New Episode"
def load_segmented_trajectories(filename):
    segments = []
    segment = []
    with open(filename, 'r') as f:
        for line in f:
            if "New Episode" in line:
                if segment:
                    segments.append(np.array(segment))
                    segment = []
            else:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                try:
                    x, y, z = map(float, parts)
                    segment.append([x - 1295, y + 3, z - 1155])
                except ValueError:
                    continue
        if segment:
            segments.append(np.array(segment))
    return segments

# Smoothing function
def smooth_trajectory(traj, sigma=13.5):
    x = gaussian_filter1d(traj[:, 0], sigma)
    y = gaussian_filter1d(traj[:, 1], sigma)
    z = gaussian_filter1d(traj[:, 2], sigma)
    return np.vstack((x, y, z)).T

# File paths
file_with_episodes = "/home/shs/colcon_ws/src/smarc2/simulation/SMARCUnityStandard/Assets/Trajectory_Data/drone_positions_runid2_test1.csv"
file_single = "/home/shs/colcon_ws/src/smarc2/simulation/SMARCUnityStandard/Assets/Trajectory_Data/drone_positions.csv"

# Load and smooth datasets
episodes_red = [smooth_trajectory(ep) for ep in load_all_episodes(file_with_episodes)]
episodes_green = [smooth_trajectory(ep) for ep in load_segmented_trajectories(file_single)]

# Setup figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# RED lines from file_with_episodes
red_lines = [ax.plot([], [], [], color='red')[0] for _ in episodes_red]
red_points = [ax.plot([], [], [], 'o', color='red')[0] for _ in episodes_red]

# GREEN lines from file_single
green_lines = [ax.plot([], [], [], color='green')[0] for _ in episodes_green]
green_points = [ax.plot([], [], [], 'o', color='green')[0] for _ in episodes_green]

# Axis settings
ax.set_xlim3d(-50, 50)
ax.set_ylim3d(-50, 50)
ax.set_zlim3d(-50, 50)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Animated Trajectories (Red: Dataset 1, Green: Dataset 2)")

# Max frame count
max_len = max(
    max(len(ep) for ep in episodes_red) if episodes_red else 0,
    max(len(ep) for ep in episodes_green) if episodes_green else 0
)

# Update function
def update(frame):
    for i, traj in enumerate(episodes_red):
        if frame < len(traj):
            red_lines[i].set_data(traj[:frame+1, 0], traj[:frame+1, 1])
            red_lines[i].set_3d_properties(traj[:frame+1, 2])
            red_points[i].set_data([traj[frame, 0]], [traj[frame, 1]])
            red_points[i].set_3d_properties([traj[frame, 2]])
    for i, traj in enumerate(episodes_green):
        if frame < len(traj):
            green_lines[i].set_data(traj[:frame+1, 0], traj[:frame+1, 1])
            green_lines[i].set_3d_properties(traj[:frame+1, 2])
            green_points[i].set_data([traj[frame, 0]], [traj[frame, 1]])
            green_points[i].set_3d_properties([traj[frame, 2]])
    return red_lines + red_points + green_lines + green_points

# Run animation
ani = FuncAnimation(fig, update, frames=range(0, max_len, 5), interval=1, blit=False)
plt.show()
