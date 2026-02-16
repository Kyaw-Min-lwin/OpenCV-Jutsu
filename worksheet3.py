import random


class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Expose position
    def get_position(self):
        return self.x, self.y

    # Perception 1: Read the Direction Label
    def perceive_direction(self, world):
        if world.in_bounds(self.x, self.y):
            # Returns "north", "east", etc.
            return world.labels[self.y][self.x]["dir"]
        return None

    # Perception 2: Read the Goal Label
    def perceive_goal(self, world):
        if world.in_bounds(self.x, self.y):
            # Returns "goal" or "empty"
            return world.labels[self.y][self.x]["type"]
        return None


class Leaf:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Expose position
    def get_position(self):
        return self.x, self.y


class World:
    def __init__(self, width, height, agent, leaf, show_labels=True):
        self.width = width
        self.height = height
        self.agent = agent
        self.leaf = leaf
        self.leaf_passable = True
        self.show_labels = show_labels

        # Initialize the grid labels
        self.labels = []
        self.set_labels()

    def set_labels(self):
        # the goal is placed in the bottom-right corner
        goal_x = self.width - 1
        goal_y = self.height - 1

        for y in range(self.height):
            row = []
            for x in range(self.width):
                # 1. Determine Direction Label
                direction = "stop"  # default
                if x < goal_x:
                    direction = "east"
                elif x > goal_x:
                    direction = "west"
                elif y < goal_y:
                    direction = "south"
                elif y > goal_y:
                    direction = "north"

                # 2. Determine Goal Label
                cell_type = "empty"
                if x == goal_x and y == goal_y:
                    cell_type = "goal"

                # Store BOTH in the field
                row.append({"dir": direction, "type": cell_type})
            self.labels.append(row)

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def display(self):
        ax, ay = self.agent.get_position()
        lx, ly = self.leaf.get_position() if self.leaf else (-1, -1)

        print(f"\n--- World State ---")
        for y in range(self.height):
            row_str = []
            for x in range(self.width):
                cell = self.labels[y][x]

                if (x, y) == (ax, ay):
                    row_str.append("A")
                elif (x, y) == (lx, ly):
                    row_str.append("L")
                elif cell["type"] == "goal":
                    row_str.append("G")
                elif self.show_labels:
                    #  Show direction first letter (e.g., e, s)
                    row_str.append(cell["dir"][0])
                else:
                    row_str.append(".")
            print(" ".join(row_str))
        print()

    # Task 2 logic (Walls bounce)
    def move_task2(self, dx, dy):
        ax, ay = self.agent.get_position()
        nx, ny = ax + dx, ay + dy
        if self.in_bounds(nx, ny):
            self.agent.x, self.agent.y = nx, ny

    # Task 3 logic (Push leaf)
    def move_task3(self, dx, dy):
        ax, ay = self.agent.get_position()
        nx, ny = ax + dx, ay + dy

        if not self.in_bounds(nx, ny):
            return

        if self.leaf and (nx, ny) == self.leaf.get_position():
            lx, ly = nx + dx, ny + dy
            if self.in_bounds(lx, ly):
                self.leaf.x, self.leaf.y = lx, ly
            else:
                self.leaf = None
                print("The leaf fell off the world!")

        self.agent.x, self.agent.y = nx, ny

    def move(self, direction):
        dirs = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

        d = direction.lower()
        if d in dirs:
            dx, dy = dirs[d]
            if self.leaf_passable:
                self.move_task2(dx, dy)
            else:
                self.move_task3(dx, dy)


# --- Execution ---
if __name__ == "__main__":
    print("Select Simulation Task:")
    print("1: Hierarchical Controller (Tasks 1, 2, 3)")
    print("2: Random Walk to Goal (Task 4)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        # Start with 5x5 world (Middle start: 2, 2)
        agent = Agent(2, 2)
        leaf = Leaf(3, 2)
        world = World(5, 5, agent, leaf, show_labels=True)

        print("\n--- Starting Hierarchical Controller (Tasks 1-3) ---")
        print("Goal is at bottom right (4, 4)")
        world.display()

        # Simulation Loop
        for step in range(10):
            print(f"Step {step + 1}:")

            # 1. Check Goal (Higher Priority)
            goal_status = agent.perceive_goal(world)
            if goal_status == "goal":
                print(">>> SUCCESS: Agent perceives the Goal label. Stopping.")
                break

            # 2. If NOT Goal, check Direction (Lower Priority)
            direction = agent.perceive_direction(world)
            print(f"Agent perceives direction: {direction}")

            # 3. Act
            world.move(direction)
            world.display()

    elif choice == "2":
        # Setup for Task 4
        agent = Agent(0, 0)  # Start top-left
        leaf = Leaf(2, 2)
        world = World(5, 5, agent, leaf, show_labels=False)

        print("\n--- Starting Random Walk (Task 4) ---")
        print("Goal is at bottom right (4, 4)")
        world.display()

        steps = 0
        max_steps = 500  # Safety break to prevent infinite loops

        # --- TASK 4 LOOP ---
        while steps < max_steps:
            steps += 1

            # 1. Check if we reached the goal
            if agent.perceive_goal(world) == "goal":
                print(f">>> SUCCESS! Reached Goal in {steps} steps.")
                break

            # 2. Choose a RANDOM direction (Ignoring floor labels)
            valid_moves = ["north", "south", "east", "west"]
            direction = random.choice(valid_moves)

            # 3. Move
            world.move(direction)

            # Print progress every 10 steps so console isn't flooded
            if steps % 10 == 0:
                print(f"Step {steps}: Random move to {direction}...")

    else:
        print("Invalid choice. Please run again and enter 1 or 2.")
