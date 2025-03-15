import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# generate random cam data with random placement and heading (aiming roughly at center with noise)
def generate_sample_data(num_cameras=6):
    cams = []
    cam_infos = []
    center = (5.0, 5.0)
    for i in range(num_cameras):
        cam_x = torch.empty(1).uniform_(0, 10).item()
        cam_y = torch.empty(1).uniform_(0, 10).item()
        base_heading = math.atan2(center[1] - cam_y, center[0] - cam_x)
        noise = torch.empty(1).uniform_(-0.1, 0.1).item()
        heading = base_heading + noise
        cam_infos.append((cam_x, cam_y, heading))
        num_peeps = torch.randint(5, 15, (1,)).item()
        if num_peeps > 0:
            detection_distance = torch.randn(num_peeps, 1) * 2 + 5  
            detection_off_x = torch.randn(num_peeps, 1)
            detection_off_y = (torch.rand(num_peeps, 1) - 0.5) * 0.2  
            detection_info = torch.cat([detection_distance, detection_off_x, detection_off_y], dim=1)
            cam_info = torch.tensor([cam_x, cam_y, heading]).repeat(num_peeps, 1)
            feed = torch.cat([cam_info, detection_info], dim=1)
        else:
            feed = torch.empty(0, 6)
        cams.append(feed)

    print(cams)
    print(cam_infos)
    return cams, cam_infos

def remove_duplicates(positions, threshold=0.5):
    unique = []
    for pos in positions:
        duplicate = False
        for u in unique:
            if torch.norm(pos - u) < threshold:
                duplicate = True
                break
        if not duplicate:
            unique.append(pos)
    return torch.stack(unique) if len(unique) > 0 else torch.empty((0, positions.shape[1]))

sample_cameras_data, camera_infos = generate_sample_data(num_cameras=6)

def compute_global_position(cam_info, peep_info):
    cam_x, cam_y, cam_heading = cam_info
    dist, off_x, off_y = peep_info
    global_angle = cam_heading + off_x
    global_x = cam_x + dist * torch.cos(global_angle)
    global_y = cam_y + dist * torch.sin(global_angle)
    global_z = off_y  
    return global_x, global_y, global_z

def localize_people(cams_data, grid_bounds, grid_resolution, kernel_size=5, sigma=1.0, dup_thresh=0.5):
    x_min, x_max, y_min, y_max = grid_bounds
    H, W = grid_resolution
    pos_list = []
    for feed in cams_data:
        if feed.shape[0] == 0:
            continue
        cam_info = feed[0, :3]
        peep_info = feed[:, 3:6]
        for i in range(feed.shape[0]):
            pos = compute_global_position(cam_info, peep_info[i])
            pos_list.append([pos[0], pos[1]])
    if len(pos_list) == 0:
        return torch.zeros((H, W))
    pos_tensor = remove_duplicates(torch.tensor(pos_list), threshold=dup_thresh)
    xs = pos_tensor[:, 0]
    ys = pos_tensor[:, 1]
    xs_norm = (xs - x_min) / (x_max - x_min) * (W - 1)
    ys_norm = (ys - y_min) / (y_max - y_min) * (H - 1)
    xs_idx = xs_norm.long()
    ys_idx = ys_norm.long()
    density = torch.zeros((H, W))
    for x, y in zip(xs_idx, ys_idx):
        if 0 <= x < W and 0 <= y < H:
            density[y, x] += 1
    def gaussian_kernel(k_size, sigma):
        ax = torch.arange(-k_size // 2 + 1, k_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kern = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kern / torch.sum(kern)
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
    density = density.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    smooth_density = F.conv2d(density, kernel, padding=pad)
    return smooth_density.squeeze(0).squeeze(0)

def get_global_positions(cams_data, dup_thresh=0.5):
    pos_list = []
    for feed in cams_data:
        if feed.shape[0] == 0:
            continue
        cam_info = feed[0, :3]
        peep_info = feed[:, 3:6]
        for i in range(feed.shape[0]):
            pos = compute_global_position(cam_info, peep_info[i])
            pos_list.append([pos[0], pos[1], pos[2]])
    if len(pos_list) == 0:
        return torch.empty((0, 3))
    pos_tensor = torch.tensor(pos_list)
    xy = pos_tensor[:, :2]
    unique_xy = remove_duplicates(xy, threshold=dup_thresh)
    unique_pos = []
    for up in unique_xy:
        idx = (torch.norm(xy - up, dim=1) < dup_thresh).nonzero()[0]
        unique_pos.append(pos_tensor[idx].squeeze(0))
    return torch.stack(unique_pos) if len(unique_pos) > 0 else torch.empty((0, 3))

if __name__ == "__main__":
    grid_bounds = (0.0, 10.0, 0.0, 10.0)
    grid_resolution = (100, 100)
    density_map = localize_people(sample_cameras_data, grid_bounds, grid_resolution, kernel_size=5, sigma=1.0, dup_thresh=0.5)
    global_positions = get_global_positions(sample_cameras_data, dup_thresh=0.5)
    
    fig1, axs1 = plt.subplots(2, 3, figsize=(12, 8))
    axs1 = axs1.flatten()
    for idx, feed in enumerate(sample_cameras_data):
        if feed.shape[0] > 0:
            peep_info = feed[:, 3:6]
            axs1[idx].scatter(peep_info[:, 1].numpy(), peep_info[:, 2].numpy(), c='blue')
        axs1[idx].set_title(f"Cam {idx+1} View")
        axs1[idx].set_xlim(-3, 3)
        axs1[idx].set_ylim(-3, 3)
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs2[0].imshow(density_map.numpy(), extent=grid_bounds, origin='lower', cmap='hot')
    axs2[0].set_title("Density Heatmap")
    axs2[0].set_xlabel("X")
    axs2[0].set_ylabel("Y")
    plt.colorbar(im0, ax=axs2[0], label="Density")
    
    axs2[1].scatter(global_positions[:, 0].numpy(), global_positions[:, 1].numpy(), c='red')
    axs2[1].set_xlim(grid_bounds[0], grid_bounds[1])
    axs2[1].set_ylim(grid_bounds[2], grid_bounds[3])
    axs2[1].set_title("2D Global Pos")
    axs2[1].set_xlabel("X")
    axs2[1].set_ylabel("Y")
    
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(global_positions[:, 0].numpy(), global_positions[:, 1].numpy(), global_positions[:, 2].numpy(), c='green')
    ax3.set_title("3D Global Pos (Flat Z with slight noise)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    
    fig4 = plt.figure(figsize=(6, 6))
    ax4 = fig4.add_subplot(111)
    cam_positions = []
    for info in camera_infos:
        cam_positions.append((info[0], info[1]))
    cam_positions_tensor = torch.tensor(cam_positions)
    ax4.scatter(cam_positions_tensor[:, 0].numpy(), cam_positions_tensor[:, 1].numpy(), c='purple', marker='s')
    for i, (x, y, heading) in enumerate(camera_infos):
        dx = math.cos(heading) * 0.5
        dy = math.sin(heading) * 0.5
        ax4.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
        ax4.text(x, y, f'Cam {i+1}', fontsize=9, color='black')
    ax4.set_title("Camera Placements")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_xlim(-1, 11)
    ax4.set_ylim(-1, 11)
    
    plt.tight_layout()
    plt.show()