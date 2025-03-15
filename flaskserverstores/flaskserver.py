from flask import Flask, request
import requests, io, base64, math, random, time, datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from langchain_community.llms import Ollama

app = Flask(__name__)

# Ollama client with short single-sentence replies.
ollama_client = Ollama(base_url="http://localhost:11434", model="llama2")

NUM_CAMERAS = 6
NUM_STORES = 5

MIN_GROUP = 8
MAX_GROUP = 20

PERIOD = 10.0
SIMULATION_DELTA = 0.05  # 3 minutes per request
simulation_time = 8.0    # start at 8:00 AM

global_cameras = []
global_stores = []
global_store_phases = []
simulated_detections = None

# Keep track of request stats.
running_stats = {
    "total_requests": 0,
    "total_delay": 0.0,
    "total_accuracy": 0.0
}

# Keep track of detection count history for each store.
store_history = {sid: [] for sid in range(NUM_STORES)}

# Generate random camera placements.
for _ in range(NUM_CAMERAS):
    x = random.uniform(0, 10)
    y = random.uniform(0, 10)
    heading = random.uniform(-math.pi, math.pi)
    global_cameras.append([x, y, heading])

# Generate random store placements.
for _ in range(NUM_STORES):
    sx = random.uniform(0, 10)
    sy = random.uniform(0, 10)
    global_stores.append([sx, sy])
    global_store_phases.append(random.uniform(0, 1))

def convert_time_to_str(sim_time):
    total_minutes = int(sim_time * 60)
    hh = total_minutes // 60 % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"

def generate_sample_data():
    global simulated_detections, simulation_time
    simulation_time += SIMULATION_DELTA
    if simulated_detections is None:
        simulated_detections = []
        for cam_x, cam_y, cam_heading in global_cameras:
            feed = []
            for store_id, (sx, sy) in enumerate(global_stores):
                dx = sx - cam_x
                dy = sy - cam_y
                distance = math.sqrt(dx*dx + dy*dy)
                angle_to_store = math.atan2(dy, dx)
                rel_angle = angle_to_store - cam_heading

                # Normalize angle to [-pi, pi].
                while rel_angle > math.pi:
                    rel_angle -= 2*math.pi
                while rel_angle < -math.pi:
                    rel_angle += 2*math.pi

                # Widen angular range to ±90° and allow distance up to 20.
                if abs(rel_angle) < math.pi/2 and distance < 20:
                    phase = global_store_phases[store_id]
                    # Activity cycles with time.
                    activity = max(0, math.sin(2*math.pi*((simulation_time / PERIOD) + phase)))
                    group_size = int(MIN_GROUP + (MAX_GROUP - MIN_GROUP)*activity)
                    for _ in range(group_size):
                        det_distance = distance + random.gauss(0, 0.2)
                        det_off_x = rel_angle + random.gauss(0, 0.05)
                        det_off_y = 0
                        feed.append([cam_x, cam_y, cam_heading, det_distance, det_off_x, det_off_y, store_id])
            # Add some random background detections.
            extra = random.randint(0, 3)
            for _ in range(extra):
                det_distance = random.gauss(5, 1)
                det_off_x = random.gauss(0, 0.5)
                feed.append([cam_x, cam_y, cam_heading, det_distance, det_off_x, 0, -1])
            simulated_detections.append(feed)
        return simulated_detections, global_cameras, global_stores
    else:
        # Slightly update existing detections each refresh.
        for feed in simulated_detections:
            for d in feed:
                d[3] += random.gauss(0, 0.05)  # distance
                d[4] += random.gauss(0, 0.01)  # off_x
                d[5] = 0                      # off_y fixed at 0

        # Potentially adjust group sizes again if in range.
        for i, (cam_x, cam_y, cam_heading) in enumerate(global_cameras):
            feed = simulated_detections[i]
            store_groups = {}
            for d in feed:
                sid = d[6]
                if sid not in store_groups:
                    store_groups[sid] = []
                store_groups[sid].append(d)
            for sid, group in store_groups.items():
                if sid < 0:
                    continue
                sx, sy = global_stores[sid]
                dx = sx - cam_x
                dy = sy - cam_y
                distance = math.sqrt(dx*dx + dy*dy)
                angle_to_store = math.atan2(dy, dx)
                rel_angle = angle_to_store - cam_heading
                while rel_angle > math.pi:
                    rel_angle -= 2*math.pi
                while rel_angle < -math.pi:
                    rel_angle += 2*math.pi
                if abs(rel_angle) < math.pi/2 and distance < 20:
                    phase = global_store_phases[sid]
                    activity = max(0, math.sin(2*math.pi*((simulation_time / PERIOD) + phase)))
                    target_size = int(MIN_GROUP + (MAX_GROUP - MIN_GROUP)*activity)
                    current_size = len(group)
                    if current_size < target_size:
                        add_count = target_size - current_size
                        for _ in range(add_count):
                            det_distance = distance + random.gauss(0, 0.2)
                            det_off_x = rel_angle + random.gauss(0, 0.05)
                            group.append([cam_x, cam_y, cam_heading, det_distance, det_off_x, 0, sid])
                    elif current_size > target_size:
                        remove_count = current_size - target_size
                        for _ in range(remove_count):
                            group.pop(random.randrange(len(group)))
        return simulated_detections, global_cameras, global_stores

def compute_accuracy(global_positions, bounds, resolution):
    x_min, x_max, y_min, y_max = bounds
    H, W = resolution
    cw = (x_max - x_min)/W
    ch = (y_max - y_min)/H
    max_error = math.sqrt((cw/2)**2 + (ch/2)**2)
    if not global_positions:
        return 0
    errors = []
    for x, y, *_ in global_positions:
        i = round((x - x_min)/(x_max - x_min)*(W-1))
        j = round((y - y_min)/(y_max - y_min)*(H-1))
        cx = x_min + (i+0.5)*cw
        cy = y_min + (j+0.5)*ch
        err = math.sqrt((x - cx)**2 + (y - cy)**2)
        errors.append(err)
    avg_err = sum(errors)/len(errors) if errors else 0
    return max(0, (1 - avg_err/max_error)*100)

def short_single_sentence_advice(store_num, count, history):
    # Summarize the detection history.
    if history:
        avg = sum(history)/len(history)
        mn = min(history)
        mx = max(history)
        hist_line = f"(History: avg={avg:.1f}, min={mn}, max={mx}.)"
    else:
        hist_line = "(No history yet.)"
    # Ask for a single-sentence reply.
    prompt = (
      f"Store {store_num} sees {count} detections now. {hist_line} "
      "Give a single-sentence recommendation on operating hours and staffing."
    )
    resp = ollama_client.invoke(prompt)
    return resp.strip()

@app.route('/')
def index():
    # Generate or update detections.
    data, cams, stores = generate_sample_data()
    payload = {
        "cams_data": data,
        "grid_bounds": [0.0, 10.0, 0.0, 10.0],
        "grid_resolution": [100, 100],
        "dup_thresh": 0.5,
        "kernel_size": 5,
        "sigma": 1.0
    }
    # Call the localization server.
    t0 = time.time()
    r = requests.post("http://localhost:5001/calculate", json=payload)
    t1 = time.time()
    lat = t1 - t0
    print(f"Request latency: {lat:.3f} seconds")

    if r.status_code != 200:
        return "Localization server error."

    # Get results.
    res = r.json()
    heatmap = res["heatmap"]
    global_positions = res["global_positions"]

    # Compute accuracy & update stats.
    acc = compute_accuracy(global_positions, payload["grid_bounds"], payload["grid_resolution"])
    running_stats["total_requests"] += 1
    running_stats["total_delay"] += lat
    running_stats["total_accuracy"] += acc
    avg_delay = running_stats["total_delay"]/running_stats["total_requests"]
    avg_acc = running_stats["total_accuracy"]/running_stats["total_requests"]
    # Current time-of-day
    time_str = convert_time_to_str(simulation_time)

    # Build store counts & update store history.
    store_counts = {sid: 0 for sid in range(NUM_STORES)}
    for feed in data:
        for d in feed:
            sid = d[6]
            if sid >= 0:
                store_counts[sid] += 1
    # Update global store_history
    for sid in range(NUM_STORES):
        store_history[sid].append(store_counts[sid])

    # Build the figure.
    fig = plt.figure(figsize=(12, 20))
    gs = plt.GridSpec(4, 1, height_ratios=[2, 2, 6, 3])

    # Row 0: Heatmap
    ax0 = fig.add_subplot(gs[0])
    import numpy as np
    hm_array = np.array(heatmap).reshape((100, 100))
    im = ax0.imshow(hm_array, extent=(0, 10, 0, 10), origin='lower', cmap='hot')
    ax0.set_title("Density Heatmap")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    # Plot store markers
    sx = [s[0] for s in stores]
    sy = [s[1] for s in stores]
    ax0.scatter(sx, sy, marker='*', s=150, c='cyan', label='Store')
    ax0.legend()
    fig.colorbar(im, ax=ax0)

    # Row 1: Actual Global Positions
    ax1 = fig.add_subplot(gs[1])
    ax1.set_title("Actual Global Positions")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    if global_positions:
        gx = [p[0] for p in global_positions]
        gy = [p[1] for p in global_positions]
        ax1.scatter(gx, gy, c='red')

    # Row 2: Camera Views
    gs_cam = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2])
    for i in range(NUM_CAMERAS):
        ax_cam = fig.add_subplot(gs_cam[i])
        feed = data[i]
        xs = [d[4] for d in feed]
        ys = [0 for _ in feed]
        ax_cam.scatter(xs, ys, c='blue', s=10)
        ax_cam.set_title(f"Cam {i+1} View")
        ax_cam.set_xlim(-math.pi/2, math.pi/2)  # expanded since we used ±90° above
        ax_cam.set_ylim(-0.1, 0.1)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

    # Row 3: Bottom row with camera placements and stats
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3])
    ax2 = fig.add_subplot(gs_bot[0])
    ax2.set_title("Camera Placements")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    cxs = [c[0] for c in cams]
    cys = [c[1] for c in cams]
    ax2.scatter(cxs, cys, c='purple', marker='s')
    for i, (cx, cy, heading) in enumerate(cams):
        dx = math.cos(heading)*0.5
        dy = math.sin(heading)*0.5
        ax2.arrow(cx, cy, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
        ax2.text(cx, cy, f"Cam {i+1}", fontsize=8, color='black')

    ax3 = fig.add_subplot(gs_bot[1])
    ax3.axis('off')
    stats_text = (
      f"Time: {time_str}\n\n"
      f"Latest Latency: {lat:.3f}s\nAccuracy: {acc:.1f}%\n\n"
      f"Running Avg Latency: {avg_delay:.3f}s\nRunning Avg Accuracy: {avg_acc:.1f}%\n\n"
      f"Current Store Counts: {store_counts}"
    )
    ax3.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    # Build the HTML
    html = f"""
    <html>
      <head>
        <title>Short Single-Sentence Advice</title>
        <meta http-equiv="refresh" content="1">
      </head>
      <body>
        <h1>Dashboard</h1>
        <p>Time: {time_str}<br>
           Latency: {lat:.3f}s<br>
           Accuracy: {acc:.1f}%<br>
           Avg Latency: {avg_delay:.3f}s<br>
           Avg Accuracy: {avg_acc:.1f}%</p>
        <p><a href="/store-advice">Store Advice Form</a></p>
        <img src="data:image/png;base64,{b64}" alt="dashboard">
      </body>
    </html>
    """
    return html

@app.route('/store-advice', methods=['GET', 'POST'])
def store_advice():
    if request.method == 'POST':
        store_number = request.form.get("store_number", type=int)
        if store_number is None or store_number < 1 or store_number > NUM_STORES:
            return "Invalid store number. <a href='/store-advice'>Go Back</a>"
        # Grab the detection count from the last iteration
        if simulated_detections is None:
            return "No data yet."
        count = 0
        for feed in simulated_detections:
            for d in feed:
                if d[6] == store_number - 1:
                    count += 1
        # Historical data
        hist = store_history.get(store_number - 1, [])
        # Single-sentence LLM advice
        advice = short_single_sentence_advice(store_number, count, hist)
        return f"<h1>Store {store_number} Advice</h1><p>{advice}</p><p><a href='/store-advice'>Back</a></p>"
    else:
        # Simple form with a dropdown for store selection
        options = "".join([f"<option value='{i+1}'>Store {i+1}</option>" for i in range(NUM_STORES)])
        return f"""
        <html>
          <head><title>Store Advice</title></head>
          <body>
            <h1>Get Single-Sentence Advice for a Store</h1>
            <form method="post">
              <label>Store:</label>
              <select name="store_number">{options}</select>
              <input type="submit" value="Get Advice">
            </form>
            <p><a href="/">Back to Dashboard</a></p>
          </body>
        </html>
        """

if __name__ == '__main__':
    app.run(port=5002)