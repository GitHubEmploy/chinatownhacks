from flask import Flask, jsonify, request
import threading
import tkinter as tk
from tkinter_sim import SimulationApp  # ensure SimulationApp has get_person_coordinates method

app = Flask(__name__)

# Create the Tk root and simulation instance in the main thread.
root = tk.Tk()
simulation_app = SimulationApp(root)

@app.route('/coords', methods=['GET'])
def get_coordinates():
    """
    Optional query parameter 'index' can be provided.
    If not provided, returns coordinates for all people.
    """
    index = request.args.get('index', default=None, type=int)
    coords = simulation_app.get_person_coordinates(index)
    return jsonify(coords)

def run_flask():
    # Run the Flask server
    app.run(host='0.0.0.0', port=5000)

# Start Flask in a daemon background thread.
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Now run the Tkinter main loop in the main thread.
root.mainloop()

