import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d


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
                x, y, z = map(float, line.strip().split(','))
                episode.append([x - 1297, y + 1, z - 1157])
        if episode:
            all_episodes.append(np.array(episode))
    return all_episodes

filename = "/home/shs/colcon_ws/src/smarc2/simulation/SMARCUnityStandard/Assets/drone_positions.csv"

episodes = load_all_episodes(filename)

from scipy.ndimage import gaussian_filter1d

def smooth_trajectory(traj, sigma=1.5):
    x = gaussian_filter1d(traj[:, 0], sigma)
    y = gaussian_filter1d(traj[:, 1], sigma)
    z = gaussian_filter1d(traj[:, 2], sigma)
    return np.vstack((x, y, z)).T

# Apply smoothing to all episodes
episodes = [smooth_trajectory(ep, sigma=1.5) for ep in episodes]


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.jet(np.linspace(0, 1, len(episodes)))
lines = [ax.plot([], [], [], color=c)[0] for c in colors]
points = [ax.plot([], [], [], 'o', color=c)[0] for c in colors]

# You can adjust the limits depending on your data
ax.set_xlim3d(-50, 50)
ax.set_ylim3d(-50, 50)
ax.set_zlim3d(-50, 50)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Animated Drone Trajectories")

max_len = max(len(e) for e in episodes)

def update(frame):
    for i, traj in enumerate(episodes):
        if frame < len(traj):
            lines[i].set_data(traj[:frame+1, 0], traj[:frame+1, 1])
            lines[i].set_3d_properties(traj[:frame+1, 2])
            points[i].set_data([traj[frame, 0]], [traj[frame, 1]])
            points[i].set_3d_properties([traj[frame, 2]])
    return lines + points


ani = FuncAnimation(fig, update, frames=range(0, max_len, 5), interval=0.5, blit=False)
plt.show()
