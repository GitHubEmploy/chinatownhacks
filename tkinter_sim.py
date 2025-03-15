import tkinter as tk
import random
import math


class Person:
    def __init__(self, canvas, x, y):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.dx = random.uniform(-2, 2)
        self.dy = random.uniform(-2, 2)
        self.group_iterations = 0
        self.cooldown = 0
        self.change_direction_counter = random.randint(50, 100)
        self.id = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="blue")

    def move(self):
        if self.group_iterations > 0:
            self.group_iterations -= 1
        elif self.cooldown > 0:
            self.cooldown -= 1
        else:
            self.change_direction_counter -= 1
            if self.change_direction_counter <= 0:
                self.dx = random.uniform(-2, 2)
                self.dy = random.uniform(-2, 2)
                self.change_direction_counter = random.randint(50, 100)

        self.x += self.dx
        self.y += self.dy

        # Keep within bounds
        if self.x < 5 or self.x > 495:
            self.dx = -self.dx
        if self.y < 5 or self.y > 495:
            self.dy = -self.dy

        self.canvas.move(self.id, self.dx, self.dy)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D View Simulations and Coordinates")

        # Full view
        tk.Label(root, text="Full View").grid(row=0, column=0)
        self.main_canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.main_canvas.grid(row=1, column=0, rowspan=3)

        # Camera views (enlarged dimensions: 600x200)
        self.cameras = []
        camera_labels = ["Cam 1", "Cam 2", "Cam 3", "Cam 4", "Cam 5", "Cam 6"]
        for i in range(6):
            tk.Label(root, text=camera_labels[i]).grid(row=(i // 3) * 2, column=1 + i % 3)
            cam_canvas = tk.Canvas(root, width=800, height=200, bg="lightgray")
            cam_canvas.grid(row=(i // 3) * 2 + 1, column=1 + i % 3)
            self.cameras.append(cam_canvas)

        self.people = []
        self.create_people(200)  # Create roughly 200 people

        self.update_simulation()

    def create_people(self, num_people):
        for _ in range(num_people):
            x = random.randint(50, 450)
            y = random.randint(50, 450)
            person = Person(self.main_canvas, x, y)
            self.people.append(person)

    def update_simulation(self):
        for person in self.people:
            person.move()

        self.check_group_movements()
        self.update_camera_views()

        self.root.after(100, self.update_simulation)  # Update frequency

    def check_group_movements(self):
        for person in self.people:
            if person.group_iterations > 0 or person.cooldown > 0:
                continue
            nearby_people = [p for p in self.people
                             if p != person and person.distance_to(p) < 20 and p.cooldown == 0]
            if nearby_people:
                group_size = random.choices([5, 8], [95, 5])[0]
                if len(nearby_people) >= group_size - 1:
                    group = nearby_people[:group_size - 1]
                    group.append(person)
                    for p in group:
                        p.group_iterations = 10
                        p.dx = person.dx
                        p.dy = person.dy
                        self.main_canvas.itemconfig(p.id, fill="red")
                else:
                    for p in nearby_people:
                        self.main_canvas.itemconfig(p.id, fill="blue")
            else:
                self.main_canvas.itemconfig(person.id, fill="blue")

        # Handle disbanding and cooldown
        for person in self.people:
            if person.group_iterations == 0 and self.main_canvas.itemcget(person.id, "fill") == "red":
                person.cooldown = 50  # Set cooldown period
                person.dx = random.uniform(-2, 2)  # Change heading
                person.dy = random.uniform(-2, 2)
                self.main_canvas.itemconfig(person.id, fill="blue")

    def update_camera_views(self):
        # Clear each camera view
        for cam_canvas in self.cameras:
            cam_canvas.delete("all")
        # Sort people by x so that they project in order without overlap.
        sorted_people = sorted(self.people, key=lambda p: p.x)
        for i, cam_canvas in enumerate(self.cameras):
            width = int(cam_canvas['width'])
            height = int(cam_canvas['height'])
            for person in sorted_people:
                # Map person's x coordinate (0 to 500) to camera canvas with margins.
                proj_x = 10 + (person.x / 500) * (width - 20)
                # Instead of a sine pattern, use a constant y (middle of the canvas),
                # giving a flat horizontal line.
                proj_y = height / 2
                color = self.main_canvas.itemcget(person.id, "fill")
                cam_canvas.create_oval(proj_x-3, proj_y-3, proj_x+3, proj_y+3, fill=color)

    def get_person_coordinates(self, index=None):
        """
        Returns the coordinates of persons.
        If index is None, returns a list of dictionaries for all persons.
        Otherwise, returns a dictionary for the person at that index (if valid).
        """
        if index is None:
            return [{'index': i, 'x': p.x, 'y': p.y} for i, p in enumerate(self.people)]
        else:
            if 0 <= index < len(self.people):
                return {'index': index, 'x': self.people[index].x, 'y': self.people[index].y}
            else:
                return None


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()