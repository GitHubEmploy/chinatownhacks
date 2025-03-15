import subprocess

client_process = subprocess.Popen([
    "/Users/mohitvarikuti/Downloads/chinahacks/.venv/bin/python",
    "/Users/mohitvarikuti/Downloads/chinahacks/flaskserverstores/flaskserver.py"
])

server_process = subprocess.Popen([
    "/Users/mohitvarikuti/Downloads/chinahacks/.venv/bin/python",
    "/Users/mohitvarikuti/Downloads/chinahacks/flaskserverstores/computationserver.py"
])

# Optional: Wait for both servers to finish
client_process.wait()
server_process.wait()