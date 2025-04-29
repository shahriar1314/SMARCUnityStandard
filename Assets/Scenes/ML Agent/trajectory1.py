import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d


def load_all_episodes(filename, skip_rate=10):  # show 1 in every 3 episodes
    all_episodes = []
    episode = []
    episode_count = 0

    with open(filename, 'r') as f:
        for line in f:
            if "New Episode" in line:
                if episode:
                    if episode_count % skip_rate == 0:
                        all_episodes.append(np.array(episode))
                    episode = []
                    episode_count += 1
            elif line.strip():
                ep_id, x, y, z = map(float, line.strip().split(','))
                ep_id = int(ep_id)
                episode.append([x - 1295, y + 3, z - 1155])
        
        # Handle last episode
        if episode and episode_count % skip_rate == 0:
            all_episodes.append(np.array(episode))

    return all_episodes


filename = "/home/shs/colcon_ws/src/smarc2/simulation/SMARCUnityStandard/Assets/Trajectory_Data/drone_positions_runid2_test1.csv"

episodes = load_all_episodes(filename)

from scipy.ndimage import gaussian_filter1d

def moving_average(data, window_size=15):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def smooth_trajectory(traj, sigma=150, window_size = 5):
    x = gaussian_filter1d(traj[:, 0], sigma)
    y = gaussian_filter1d(traj[:, 1], sigma)
    z = gaussian_filter1d(traj[:, 2], sigma)

    # x = moving_average(traj[:, 0], window_size)
    # y = moving_average(traj[:, 1], window_size)
    # z = moving_average(traj[:, 2], window_size)
    return np.vstack((x, y, z)).T

# Apply smoothing to all episodes
episodes = [smooth_trajectory(ep, sigma=13.5) for ep in episodes]


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.jet(np.linspace(0, 1, len(episodes)))
lines = [ax.plot([], [], [], color=c)[0] for c in colors]
points = [ax.plot([], [], [], 'o', color=c)[0] for c in colors]

# You can adjust the limits depending on your data
ax.set_xlim3d(-5, 15)
ax.set_ylim3d(-5, 15)
ax.set_zlim3d(-5, 15)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Animated Drone Trajectories")
ax.set_title(f"Episode Range: 1-10 ", fontsize=10, loc='right', pad=10)


max_len = max(len(e) for e in episodes)

def update(frame):
    for i, traj in enumerate(episodes):
        if frame < len(traj):
            lines[i].set_data(traj[:frame+1, 0], traj[:frame+1, 1])
            lines[i].set_3d_properties(traj[:frame+1, 2])
            points[i].set_data([traj[frame, 0]], [traj[frame, 1]])
            points[i].set_3d_properties([traj[frame, 2]])
    return lines + points

# # Add water volume by stacking transparent XY planes from z=-50 to z=0
# z_levels = np.linspace(-50, 0, 20)  # Adjust number of slices for smoothness
# x = np.linspace(-50, 50, 20)
# y = np.linspace(-50, 50, 20)
# x, y = np.meshgrid(x, y)

# for z in z_levels:
#     ax.plot_surface(
#         x, y, np.full_like(x, z),
#         color='lightblue', alpha=0.05, linewidth=0, antialiased=False
#     )


ani = FuncAnimation(fig, update, frames=range(0, max_len, 5), interval=1, blit=False)
plt.show()
