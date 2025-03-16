from flask import Flask, request
import requests, io, base64, math, random, time, datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from langchain_community.llms import Ollama
import numpy as np

app = Flask(__name__)

# Instantiate an Ollama client with the model.
ollama_client = Ollama(base_url="http://localhost:11434", model="llama2")

NUM_CAMERAS = 6
NUM_STORES = 5

MIN_GROUP = 8
MAX_GROUP = 20
PERIOD = 10.0
SIMULATION_DELTA = 0.05  # 3 minutes per request
simulation_time = 8.0    # Start at 8:00 AM

# Global running stats.
running_stats = {
    "total_requests": 0,
    "total_delay": 0.0,
    "total_accuracy": 0.0
}

# Global history for store detection counts.
store_history = {sid: [] for sid in range(NUM_STORES)}

# Fixed placements.
global_cameras = []      # Each: [cam_x, cam_y, cam_heading]
global_stores = []       # Each: [store_x, store_y]
global_store_phases = [] # Per-store phase

for _ in range(NUM_CAMERAS):
    x = random.uniform(0, 10)
    y = random.uniform(0, 10)
    heading = random.uniform(-math.pi, math.pi)
    global_cameras.append([x, y, heading])

for _ in range(NUM_STORES):
    sx = random.uniform(0, 10)
    sy = random.uniform(0, 10)
    global_stores.append([sx, sy])
    global_store_phases.append(random.uniform(0, 1))

# Global simulation state.
simulated_detections = None

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
                while rel_angle > math.pi:
                    rel_angle -= 2*math.pi
                while rel_angle < -math.pi:
                    rel_angle += 2*math.pi
                if abs(rel_angle) < math.pi/2 and distance < 20:
                    phase = global_store_phases[store_id]
                    activity = max(0, math.sin(2*math.pi*(simulation_time/PERIOD + phase)))
                    group_size = int(MIN_GROUP + (MAX_GROUP - MIN_GROUP)*activity)
                    for _ in range(group_size):
                        det_distance = distance + random.gauss(0, 0.2)
                        det_off_x = rel_angle + random.gauss(0, 0.05)
                        det_off_y = 0  # No vertical variation
                        feed.append([cam_x, cam_y, cam_heading, det_distance, det_off_x, det_off_y, store_id])
            extra = random.randint(0, 3)
            for _ in range(extra):
                det_distance = random.gauss(5, 1)
                det_off_x = random.gauss(0, 0.5)
                det_off_y = 0
                feed.append([cam_x, cam_y, cam_heading, det_distance, det_off_x, det_off_y, -1])
            simulated_detections.append(feed)
        return simulated_detections, global_cameras, global_stores
    else:
        for feed in simulated_detections:
            for d in feed:
                d[3] += random.gauss(0, 0.05)
                d[4] += random.gauss(0, 0.01)
                d[5] = 0
        for i, (cam_x, cam_y, cam_heading) in enumerate(global_cameras):
            feed = simulated_detections[i]
            store_groups = {}
            for d in feed:
                sid = d[6]
                if sid not in store_groups:
                    store_groups[sid] = []
                store_groups[sid].append(d)
            for sid, group in store_groups.items():
                if sid == -1:
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
                    activity = max(0, math.sin(2*math.pi*(simulation_time/PERIOD + phase)))
                    target_size = int(MIN_GROUP + (MAX_GROUP - MIN_GROUP)*activity)
                    current_size = len(group)
                    if current_size < target_size:
                        for _ in range(target_size - current_size):
                            det_distance = distance + random.gauss(0, 0.2)
                            det_off_x = rel_angle + random.gauss(0, 0.05)
                            det_off_y = 0
                            feed.append([cam_x, cam_y, cam_heading, det_distance, det_off_x, det_off_y, sid])
                    elif current_size > target_size:
                        for _ in range(current_size - target_size):
                            group.pop(random.randrange(len(group)))
        return simulated_detections, global_cameras, global_stores

def compute_accuracy(global_positions, bounds, resolution):
    x_min, x_max, y_min, y_max = bounds
    H, W = resolution
    cw = (x_max - x_min) / W
    ch = (y_max - y_min) / H
    max_error = math.sqrt((cw/2)**2 + (ch/2)**2)
    if not global_positions:
        return 0
    errs = []
    for x, y, _ in global_positions:
        i = round((x - x_min)/(x_max - x_min)*(W-1))
        j = round((y - y_min)/(y_max - y_min)*(H-1))
        cx = x_min + (i+0.5)*cw
        cy = y_min + (j+0.5)*ch
        errs.append(math.sqrt((x-cx)**2 + (y-cy)**2))
    avg_err = sum(errs)/len(errs)
    return max(0, (1 - avg_err/max_error)*100)

def get_detailed_store_advice(store_number, detection_count, history):
    # Summarize history.
    if history:
        avg = sum(history)/len(history)
        mn = min(history)
        mx = max(history)
        hist_summary = f"Average: {avg:.1f}, Min: {mn}, Max: {mx}."
    else:
        hist_summary = "No historical data available."
    # Get store location.
    store_loc = global_stores[store_number - 1]
    # Use overall running stats.
    overall_stats = (f"Overall running average accuracy: {running_stats['total_accuracy']/running_stats['total_requests']:.1f}%, "
                     f"latency: {running_stats['total_delay']/running_stats['total_requests']:.3f}s.") if running_stats["total_requests"] > 0 else ""
    prompt = (
        f"You are a retail operations expert with detailed context. "
        f"Store {store_number} located at ({store_loc[0]:.2f}, {store_loc[1]:.2f}) is currently receiving {detection_count} detections. "
        f"Historical data for this store: {hist_summary} Current simulated time: {convert_time_to_str(simulation_time)}. "
        f"{overall_stats} Based on all this data, provide a detailed single-sentence recommendation on optimal operating hours and staffing."
    )
    response = ollama_client.invoke(prompt)
    return response.strip()

@app.route('/', methods=['GET'])
def index():
    data, cams, stores = generate_sample_data()
    payload = {
        "cams_data": data,
        "grid_bounds": [0.0, 10.0, 0.0, 10.0],
        "grid_resolution": [100, 100],
        "dup_thresh": 0.5,
        "kernel_size": 5,
        "sigma": 1.0
    }
    t0 = time.time()
    r = requests.post("http://localhost:5001/calculate", json=payload)
    t1 = time.time()
    lat = t1 - t0
    print(f"Request latency: {lat:.3f} seconds")
    
    if r.status_code != 200:
        return "Error in localization server."
    
    res = r.json()
    heatmap = res["heatmap"]
    global_positions = res["global_positions"]
    acc = compute_accuracy(global_positions, payload["grid_bounds"], payload["grid_resolution"])
    
    running_stats["total_requests"] += 1
    running_stats["total_delay"] += lat
    running_stats["total_accuracy"] += acc
    avg_delay = running_stats["total_delay"]/running_stats["total_requests"]
    avg_acc = running_stats["total_accuracy"]/running_stats["total_requests"]
    
    time_str = convert_time_to_str(simulation_time)
    
    # Update store history.
    current_counts = {sid: 0 for sid in range(NUM_STORES)}
    for feed in data:
        for d in feed:
            sid = d[6]
            if sid >= 0:
                current_counts[sid] += 1
    for sid in range(NUM_STORES):
        store_history[sid].append(current_counts.get(sid, 0))
    
    fig = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 2])
    
    plt.style.use('dark_background')  # Enable dark theme for plots
    
    # Row 0: Density Heatmap with Store Markers
    ax0 = fig.add_subplot(gs[0])
    hm_array = np.array(heatmap).reshape((100, 100))

    # Adjust the extent to correctly center the heatmap
    x_min, x_max, y_min, y_max = 0.0, 10.0, 0.0, 10.0  # Ensure grid bounds are correctly used
    im = ax0.imshow(hm_array, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='hot', aspect='auto')

    ax0.set_title("Density Heatmap", color='white')
    ax0.set_xlabel("X", color='white')
    ax0.set_ylabel("Y", color='white')

    # Plot stores as markers
    sx = [s[0] for s in stores]
    sy = [s[1] for s in stores]
    ax0.scatter(sx, sy, marker='*', s=150, c='cyan', label='Store')

    # Center the visualization
    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_min, y_max)

    ax0.legend()
    fig.colorbar(im, ax=ax0)
    
    # Row 1: Camera Views
    gs_cam = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1])
    for i in range(NUM_CAMERAS):
        ax_cam = fig.add_subplot(gs_cam[i])
        feed = data[i]
        xs = [d[4] for d in feed]
        ys = [0 for _ in feed]
        ax_cam.scatter(xs, ys, c='cyan', s=10)
        ax_cam.set_title(f"Cam {i+1} View", color='white')
        ax_cam.set_xlim(-math.pi/2, math.pi/2)
        ax_cam.set_ylim(-0.1, 0.1)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
    
    plt.tight_layout()
    
    # Save plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    # Updated HTML with dark theme
    html = f"""
    <html>
      <head>
        <title>Dashboard</title>
        <meta http-equiv="refresh" content="0.00001">
        <style>
          body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #1a1a1a;
            color: #ffffff;
            min-height: 100vh;
          }}
          .container {{
            max-width: 1000px;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
          }}
          h1 {{
            font-size: 2.5rem;
            font-weight: 300;
            margin: 20px 0;
            color: #ffffff;
          }}
          .stats-box {{
            background: #2d2d2d;
            border-radius: 16px;
            padding: 20px 40px;
            text-align: center;
            width: fit-content;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          }}
          .dashboard-image {{
            width: 100%;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
          }}
          .button-container {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
          }}
          .button {{
            background: #0066cc;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 12px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
          }}
          .button:hover {{
            background: #0052a3;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          }}
          .time {{
            font-size: 1.2rem;
            margin-bottom: 8px;
          }}
          .latency {{
            font-size: 1rem;
            opacity: 0.9;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Dashboard</h1>
          <div class="stats-box">
            <div class="time">Time: {time_str}</div>
            <div class="latency">Latency: {lat:.3f}s</div>
            <div class="button-container">
              <a href="/store-advice" class="button">Get Store Advice</a>
            </div>
          </div>
          <div class="dashboard-image">
            <img src="data:image/png;base64,{b64}" alt="Dashboard" style="width: 100%; display: block;">
          </div>
        </div>
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
        count = 0
        if simulated_detections is not None:
            for feed in simulated_detections:
                for d in feed:
                    if d[6] == store_number - 1:
                        count += 1
        hist = store_history.get(store_number - 1, [])
        advice = get_detailed_store_advice(store_number, count, hist)
        return f"""
        <html>
          <head>
            <title>Store Advice</title>
            <style>
              body {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #1a1a1a;
                color: #ffffff;
              }}
              .container {{
                max-width: 800px;
                width: 100%;
                text-align: center;
              }}
              .advice-box {{
                background: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
              }}
              a {{
                display: inline-block;
                background: #0066cc;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px 5px;
              }}
              a:hover {{
                background: #0052a3;
              }}
            </style>
          </head>
          <body>
            <div class="container">
              <h1>Advice for Store {store_number}</h1>
              <div class="advice-box">
                <p>{advice}</p>
              </div>
              <div>
                <a href="/store-advice">Back</a>
                <a href="/">Back to Dashboard</a>
              </div>
            </div>
          </body>
        </html>
        """
    else:
        options = "".join([f"<option value='{i+1}'>Store {i+1}</option>" for i in range(NUM_STORES)])
        return f"""
        <html>
          <head>
            <title>Store Advice</title>
            <style>
              body {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #1a1a1a;
                color: #ffffff;
              }}
              .container {{
                max-width: 600px;
                width: 100%;
                text-align: center;
              }}
              .form-box {{
                background: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
              }}
              select {{
                width: 200px;
                padding: 8px;
                margin: 10px 0;
                border-radius: 4px;
                border: 1px solid #444;
                background: #333;
                color: white;
              }}
              input[type="submit"] {{
                background: #0066cc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 0;
              }}
              input[type="submit"]:hover {{
                background: #0052a3;
              }}
              a {{
                display: inline-block;
                background: #0066cc;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px 0;
              }}
              a:hover {{
                background: #0052a3;
              }}
            </style>
          </head>
          <body>
            <div class="container">
              <h1>Get Store Advice</h1>
              <div class="form-box">
                <form method="post">
                  <label>Select Store:</label><br>
                  <select name="store_number">{options}</select><br>
                  <input type="submit" value="Get Advice">
                </form>
              </div>
              <a href="/">Back to Dashboard</a>
            </div>
          </body>
        </html>
        """

if __name__ == '__main__':
    app.run(port=5002)
