from flask import Flask
import requests, io, base64, math, random, time
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global stats to keep a running average
running_stats = {
    "total_requests": 0,
    "total_delay": 0.0,
    "total_accuracy": 0.0
}

def generate_sample_data(num_cameras=6):
    cams = []
    cam_infos = []  # store camera placements
    for i in range(num_cameras):
        num_peeps = random.randint(5, 15)
        cam_x = random.uniform(0, 10)
        cam_y = random.uniform(0, 10)
        base_heading = math.atan2(5.0 - cam_y, 5.0 - cam_x)  # aim roughly at center (5,5)
        noise = random.uniform(-0.1, 0.1)
        heading = base_heading + noise
        cam_infos.append([cam_x, cam_y, heading])
        feed = []
        for _ in range(num_peeps):
            detection_distance = random.gauss(5, 2)
            detection_off_x = random.gauss(0, 1)
            detection_off_y = (random.random() - 0.5) * 0.2
            detection = [cam_x, cam_y, heading, detection_distance, detection_off_x, detection_off_y]
            feed.append(detection)
        cams.append(feed)
        
    return cams, cam_infos

def compute_accuracy(global_positions, grid_bounds, grid_resolution):
    # Compute average error between each global position and its grid cell center.
    x_min, x_max, y_min, y_max = grid_bounds
    H, W = grid_resolution
    cell_width = (x_max - x_min) / W
    cell_height = (y_max - y_min) / H
    max_error = math.sqrt((cell_width/2)**2 + (cell_height/2)**2)
    errors = []
    for pos in global_positions:
        x, y = pos[0], pos[1]
        # Determine grid indices (using rounding)
        i = round((x - x_min) / (x_max - x_min) * (W - 1))
        j = round((y - y_min) / (y_max - y_min) * (H - 1))
        cell_center_x = x_min + (i + 0.5) * cell_width
        cell_center_y = y_min + (j + 0.5) * cell_height
        error = math.sqrt((x - cell_center_x)**2 + (y - cell_center_y)**2)
        errors.append(error)
    avg_error = sum(errors) / len(errors) if errors else 0
    accuracy = max(0, (1 - avg_error / max_error) * 100)
    return 100-accuracy

@app.route('/')
def index():
    sample_data, cam_infos = generate_sample_data(num_cameras=6)
    payload = {
        "cams_data": sample_data,
        "grid_bounds": [0.0, 10.0, 0.0, 10.0],
        "grid_resolution": [100, 100],
        "dup_thresh": 0.5,
        "kernel_size": 5,
        "sigma": 1.0
    }
    start_time = time.time()
    response = requests.post("http://localhost:5001/calculate", json=payload)
    end_time = time.time()
    latency = end_time - start_time
    print(f"Request latency: {latency:.3f} seconds")
    
    if response.status_code != 200:
        return "Error in localization server"
    data = response.json()
    heatmap = data.get("heatmap")
    global_positions = data.get("global_positions")
    accuracy = compute_accuracy(global_positions, payload["grid_bounds"], payload["grid_resolution"]) if global_positions else 0
    
    # Update running averages
    running_stats["total_requests"] += 1
    running_stats["total_delay"] += latency
    running_stats["total_accuracy"] += accuracy
    avg_delay = running_stats["total_delay"] / running_stats["total_requests"]
    avg_accuracy = running_stats["total_accuracy"] / running_stats["total_requests"]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Density Heatmap
    im = axs[0,0].imshow(heatmap, extent=(0.0, 10.0, 0.0, 10.0), origin='lower', cmap='hot')
    axs[0,0].set_title("Density Heatmap")
    axs[0,0].set_xlabel("X")
    axs[0,0].set_ylabel("Y")
    fig.colorbar(im, ax=axs[0,0])
    
    # Global Positions Scatter
    if global_positions:
        xs = [pos[0] for pos in global_positions]
        ys = [pos[1] for pos in global_positions]
        axs[0,1].scatter(xs, ys, c='red')
    axs[0,1].set_title("Global Positions")
    axs[0,1].set_xlabel("X")
    axs[0,1].set_ylabel("Y")
    axs[0,1].set_xlim(0,10)
    axs[0,1].set_ylim(0,10)
    
    # Camera Placements Map
    cam_xs = [info[0] for info in cam_infos]
    cam_ys = [info[1] for info in cam_infos]
    axs[1,0].scatter(cam_xs, cam_ys, c='purple', marker='s')
    for i, info in enumerate(cam_infos):
        x, y, heading = info
        dx = math.cos(heading) * 0.5
        dy = math.sin(heading) * 0.5
        axs[1,0].arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
        axs[1,0].text(x, y, f'Cam {i+1}', fontsize=9, color='black')
    axs[1,0].set_title("Camera Placements")
    axs[1,0].set_xlabel("X")
    axs[1,0].set_ylabel("Y")
    axs[1,0].set_xlim(0,10)
    axs[1,0].set_ylim(0,10)
    
    # Additional Plot: Running Averages
    axs[1,1].axis('off')
    stats_text = f"Latest Request:\nLatency: {latency:.3f} s\nAccuracy: {accuracy:.1f}%\n\n" \
                 f"Running Average:\nLatency: {avg_delay:.3f} s\nAccuracy: {avg_accuracy:.1f}%"
    axs[1,1].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    
    html = f"""
    <html>
      <head><title>Heatmap Visualization</title></head>
      <body>
        <h1>Heatmap, Global Positions, and Camera Placements</h1>
        <p>Latest Request Latency: {latency:.3f} seconds<br>
           Latest Accuracy: {accuracy:.1f}%<br>
           Running Average Latency: {avg_delay:.3f} seconds<br>
           Running Average Accuracy: {avg_accuracy:.1f}%</p>
        <img src="data:image/png;base64,{image_base64}" alt="Visualization">
      </body>
    </html>
    """
    return html

if __name__ == '__main__':
    app.run(port=5000)