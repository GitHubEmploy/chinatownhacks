from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import math

app = Flask(__name__)

def remove_duplicates(positions, threshold=0.5):
    unique = []
    for pos in positions:
        duplicate = False
        for u in unique:
            if torch.norm(torch.tensor(pos) - torch.tensor(u)) < threshold:
                duplicate = True
                break
        if not duplicate:
            unique.append(pos)
    return unique

def compute_global_position(cam_info, peep_info):
    cam_x, cam_y, cam_heading = cam_info
    dist, off_x, off_y = peep_info
    global_angle = cam_heading + off_x
    global_x = cam_x + dist * math.cos(global_angle)
    global_y = cam_y + dist * math.sin(global_angle)
    global_z = off_y
    return [global_x, global_y, global_z]

def localize_people(cams_data, grid_bounds, grid_resolution, kernel_size=5, sigma=1.0, dup_thresh=0.5):
    x_min, x_max, y_min, y_max = grid_bounds
    H, W = grid_resolution
    pos_list = []
    for feed in cams_data:
        if len(feed) == 0:
            continue
        cam_info = feed[0][:3]
        for peep in feed:
            pos = compute_global_position(cam_info, peep[3:6])
            pos_list.append([pos[0], pos[1]])
    pos_list = remove_duplicates(pos_list, threshold=dup_thresh)
    density = torch.zeros((H, W))
    for pos in pos_list:
        x, y = pos
        xs_norm = (x - x_min) / (x_max - x_min) * (W - 1)
        ys_norm = (y - y_min) / (y_max - y_min) * (H - 1)
        x_idx = int(round(xs_norm))
        y_idx = int(round(ys_norm))
        if 0 <= x_idx < W and 0 <= y_idx < H:
            density[y_idx, x_idx] += 1
    ax = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kern = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kern = kern / torch.sum(kern)
    kern = kern.unsqueeze(0).unsqueeze(0)
    density = density.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    smooth_density = F.conv2d(density, kern, padding=pad)
    return smooth_density.squeeze().tolist()

def get_global_positions(cams_data, dup_thresh=0.5):
    pos_list = []
    for feed in cams_data:
        if len(feed) == 0:
            continue
        cam_info = feed[0][:3]
        for peep in feed:
            pos = compute_global_position(cam_info, peep[3:6])
            pos_list.append(pos)
    unique = []
    for pos in pos_list:
        xy = pos[:2]
        duplicate = False
        for u in unique:
            if math.sqrt((xy[0]-u[0])**2 + (xy[1]-u[1])**2) < dup_thresh:
                duplicate = True
                break
        if not duplicate:
            unique.append(pos)
    return unique

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    cams_data = data.get("cams_data", [])
    grid_bounds = data.get("grid_bounds", [0.0, 10.0, 0.0, 10.0])
    grid_resolution = data.get("grid_resolution", [100, 100])
    dup_thresh = data.get("dup_thresh", 0.5)
    kernel_size = data.get("kernel_size", 5)
    sigma = data.get("sigma", 1.0)
    
    heatmap = localize_people(cams_data, grid_bounds, grid_resolution, kernel_size, sigma, dup_thresh)
    global_positions = get_global_positions(cams_data, dup_thresh)
    return jsonify({
        "heatmap": heatmap,
        "global_positions": global_positions
    })

if __name__ == '__main__':
    app.run(port=5001)
