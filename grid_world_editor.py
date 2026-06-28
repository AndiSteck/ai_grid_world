"""
Grid World Editor for JEPA/VLA experiments.
10x10 grid world with 16-color VGA palette, tkinter editor, matplotlib visualization.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrow, Circle
import math

# ----------------------------
# VGA 16-color palette (RGB)
# ----------------------------
VGA_PALETTE = [
    (0, 0, 0),        # 0  Black       - Wall
    (0, 0, 170),      # 1  Blue
    (0, 170, 0),      # 2  Green       - Goal
    (0, 170, 170),    # 3  Cyan
    (170, 0, 0),      # 4  Red
    (170, 0, 170),    # 5  Magenta
    (170, 85, 0),     # 6  Brown       - Door (closed)
    (170, 170, 170),  # 7  Light Gray  - Door (open)
    (85, 85, 85),     # 8  Dark Gray
    (85, 85, 255),    # 9  Light Blue
    (85, 255, 85),    # 10 Light Green
    (85, 255, 255),   # 11 Light Cyan
    (255, 85, 85),    # 12 Light Red
    (255, 85, 255),   # 13 Light Magenta
    (255, 255, 85),   # 14 Yellow      - Key
    (255, 255, 255),  # 15 White       - Empty
]

# Cell type constants
CELL_WALL = 0       # Black
CELL_GOAL = 2       # Green
CELL_DOOR_CLOSED = 6  # Brown
CELL_DOOR_OPEN = 7    # Light Gray
CELL_KEY = 14       # Yellow
CELL_EMPTY = 15     # White

# Direction constants
DIR_NORTH = 0  # -Y
DIR_EAST = 1   # +X
DIR_SOUTH = 2  # +Y
DIR_WEST = 3   # -X

DIR_DX = [0, 1, 0, -1]
DIR_DY = [-1, 0, 1, 0]

TOOL_NAMES = {
    CELL_EMPTY: "Empty",
    CELL_WALL: "Wall",
    CELL_DOOR_CLOSED: "Door",
    CELL_KEY: "Key",
    CELL_GOAL: "Goal",
}


# ----------------------------
# Grid World Model
# ----------------------------
class GridWorld:
    def __init__(self):
        self._width = 10
        self._height = 10
        self._grid = np.full((self._height, self._width), CELL_EMPTY, dtype=np.uint8)
        # Robot state
        self._robot_x = 0
        self._robot_y = 0
        self._robot_dir = DIR_EAST  # facing right
        self._robot_inventory = None  # None or cell type value

    def reset(self):
        self._grid = np.full((self._height, self._width), CELL_EMPTY, dtype=np.uint8)
        self._robot_x = 0
        self._robot_y = 0
        self._robot_dir = DIR_EAST
        self._robot_inventory = None

    def set_start_pose(self, x, y, direction):
        """Set robot starting pose."""
        self._robot_x = x
        self._robot_y = y
        self._robot_dir = direction

    def on_goal(self):
        """Return True if robot is currently on the goal cell."""
        return self._grid[self._robot_y, self._robot_x] == CELL_GOAL

    def _in_bounds(self, x, y):
        return 0 <= x < self._width and 0 <= y < self._height

    def _front_cell(self):
        """Return (x, y) of cell in front of robot."""
        fx = self._robot_x + DIR_DX[self._robot_dir]
        fy = self._robot_y + DIR_DY[self._robot_dir]
        return fx, fy

    def _can_move_to(self, x, y):
        if not self._in_bounds(x, y):
            return False
        cell = self._grid[y, x]
        if cell == CELL_WALL or cell == CELL_DOOR_CLOSED:
            return False
        return True

    def _move_forward(self):
        fx, fy = self._front_cell()
        if self._can_move_to(fx, fy):
            self._robot_x = fx
            self._robot_y = fy
        self._update_doors()

    def _move_backward(self):
        # Move opposite to facing direction
        bx = self._robot_x - DIR_DX[self._robot_dir]
        by = self._robot_y - DIR_DY[self._robot_dir]
        if self._can_move_to(bx, by):
            self._robot_x = bx
            self._robot_y = by
        self._update_doors()

    def _turn_left(self):
        self._robot_dir = (self._robot_dir - 1) % 4
        self._update_doors()

    def _turn_right(self):
        self._robot_dir = (self._robot_dir + 1) % 4
        self._update_doors()

    def _pickup(self):
        """Pick up object in front of robot if inventory is empty."""
        fx, fy = self._front_cell()
        if not self._in_bounds(fx, fy):
            return False
        cell = self._grid[fy, fx]
        if self._robot_inventory is None and cell == CELL_KEY:
            self._robot_inventory = cell
            self._grid[fy, fx] = CELL_EMPTY
            return True
        return False

    def _use_object(self):
        """Use held object on cell in front of robot."""
        fx, fy = self._front_cell()
        if not self._in_bounds(fx, fy):
            return False
        cell = self._grid[fy, fx]
        if self._robot_inventory == CELL_KEY and cell == CELL_DOOR_CLOSED:
            self._grid[fy, fx] = CELL_DOOR_OPEN
            return True
        return False

    def _drop_object(self):
        """Drop held object onto cell in front of robot if that cell is empty."""
        if self._robot_inventory is None:
            return False
        fx, fy = self._front_cell()
        if not self._in_bounds(fx, fy):
            return False
        if self._grid[fy, fx] != CELL_EMPTY:
            return False
        self._grid[fy, fx] = self._robot_inventory
        self._robot_inventory = None
        return True

    # --- Public API ---
    # Action indices:
    #   0=forward, 1=backward, 2=turn_left, 3=turn_right,
    #   4=pickup, 5=use_object, 6=drop
    ACTIONS = ["forward", "backward", "turn_left", "turn_right",
               "pickup", "use_object", "drop"]

    def do_action(self, action_id):
        """Execute a robot action by integer index. Returns result of action."""
        handlers = [
            self._move_forward,
            self._move_backward,
            self._turn_left,
            self._turn_right,
            self._pickup,
            self._use_object,
            self._drop_object,
        ]
        if not 0 <= action_id < len(handlers):
            raise ValueError(f"action_id must be 0-{len(handlers)-1}, got {action_id}")
        return handlers[action_id]()

    def _update_doors(self):
        """Close open doors if robot distance > 1."""
        for y in range(self._height):
            for x in range(self._width):
                if self._grid[y, x] == CELL_DOOR_OPEN:
                    dist = abs(x - self._robot_x) + abs(y - self._robot_y)
                    if dist > 1:
                        self._grid[y, x] = CELL_DOOR_CLOSED

    def get_observation(self):
        """Return observation: grid values (0-15), robot pose (x, y, yaw), and inventory."""
        return {
            "grid": self._grid.copy(),
            "pose": (self._robot_x, self._robot_y, self._robot_dir),
            "inventory": self._robot_inventory,  # None or cell type (e.g. 14 = key)
        }

    def save_png(self, filepath):
        """Save world as 16-color VGA indexed PNG."""
        img = Image.new("P", (self._width, self._height))
        # Set palette
        palette = []
        for r, g, b in VGA_PALETTE:
            palette.extend([r, g, b])
        # Pad to 256 colors
        palette.extend([0] * (768 - len(palette)))
        img.putpalette(palette)
        # Set pixels
        for y in range(self._height):
            for x in range(self._width):
                img.putpixel((x, y), int(self._grid[y, x]))
        img.save(filepath)

    def load_png(self, filepath):
        """Load world from indexed PNG."""
        img = Image.open(filepath)
        img = img.convert("P")
        if img.size != (self._width, self._height):
            messagebox.showerror("Error", f"Image must be {self._width}x{self._height}")
            return False
        for y in range(self._height):
            for x in range(self._width):
                val = img.getpixel((x, y))
                self._grid[y, x] = min(val, 15)
        return True


# ----------------------------
# Editor GUI
# ----------------------------
class GridWorldEditor:
    def __init__(self):
        self.world = GridWorld()
        self.current_tool = CELL_WALL
        self.current_file = None
        self.drawing = False

        # Main window
        self.root = tk.Tk()
        self.root.title("Grid World Editor")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Menu bar
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self._new_world)
        filemenu.add_command(label="Load...", command=self._load_world)
        filemenu.add_command(label="Save", command=self._save_world)
        filemenu.add_command(label="Save As...", command=self._save_world_as)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # Main layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - tools
        tool_frame = tk.Frame(main_frame, width=192, padx=5, pady=5)
        tool_frame.pack(side=tk.LEFT, fill=tk.Y)
        tool_frame.pack_propagate(False)

        tk.Label(tool_frame, text="Draw Tools", font=("Arial", 10, "bold")).pack(pady=(5, 10))

        self.tool_buttons = {}
        for cell_type, name in TOOL_NAMES.items():
            color = "#{:02x}{:02x}{:02x}".format(*VGA_PALETTE[cell_type])
            fg = "white" if cell_type in (CELL_WALL,) else "black"
            btn = tk.Button(
                tool_frame, text=name, width=12,
                bg=color, fg=fg, relief=tk.RAISED,
                command=lambda ct=cell_type: self._select_tool(ct)
            )
            btn.pack(pady=2)
            self.tool_buttons[cell_type] = btn

        tk.Label(tool_frame, text="", height=1).pack()
        tk.Label(tool_frame, text="Robot", font=("Arial", 10, "bold")).pack(pady=(5, 5))

        tk.Button(tool_frame, text="Place Robot", width=12,
                  command=self._place_robot_mode).pack(pady=2)

        tk.Label(tool_frame, text="", height=1).pack()
        tk.Label(tool_frame, text="Actions", font=("Arial", 10, "bold")).pack(pady=(5, 5))

        tk.Button(tool_frame, text="Forward", width=12,
                  command=self._action_forward).pack(pady=1)
        tk.Button(tool_frame, text="Backward", width=12,
                  command=self._action_backward).pack(pady=1)
        tk.Button(tool_frame, text="Turn Left", width=12,
                  command=self._action_turn_left).pack(pady=1)
        tk.Button(tool_frame, text="Turn Right", width=12,
                  command=self._action_turn_right).pack(pady=1)
        tk.Button(tool_frame, text="Pickup", width=12,
                  command=self._action_pickup).pack(pady=1)
        tk.Button(tool_frame, text="Use Object", width=12,
                  command=self._action_use).pack(pady=1)
        tk.Button(tool_frame, text="Drop", width=12,
                  command=self._action_drop).pack(pady=1)

        # Inventory label
        tk.Label(tool_frame, text="", height=1).pack()
        self.inventory_label = tk.Label(tool_frame, text="Inventory: empty",
                                        font=("Arial", 9))
        self.inventory_label.pack(pady=2)

        # Observation button
        tk.Button(tool_frame, text="Print Obs", width=12,
                  command=self._print_observation).pack(pady=5)

        # Right panel - matplotlib canvas
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Mouse events on canvas
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

        self._place_robot = False
        self._highlight_tool()
        self._render()

    def run(self):
        self.root.mainloop()

    # --- Tool selection ---
    def _select_tool(self, tool):
        self._place_robot = False
        self.current_tool = tool
        self._highlight_tool()

    def _place_robot_mode(self):
        self._place_robot = True
        # Un-highlight draw tools
        for ct, btn in self.tool_buttons.items():
            btn.config(relief=tk.RAISED)

    def _highlight_tool(self):
        for ct, btn in self.tool_buttons.items():
            if ct == self.current_tool and not self._place_robot:
                btn.config(relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)

    # --- Canvas mouse events ---
    def _canvas_to_grid(self, event):
        if event.xdata is None or event.ydata is None:
            return None, None
        gx = int(event.xdata)
        gy = int(event.ydata)
        if 0 <= gx < self.world._width and 0 <= gy < self.world._height:
            return gx, gy
        return None, None

    def _on_canvas_press(self, event):
        if event.button != 1:
            return
        gx, gy = self._canvas_to_grid(event)
        if gx is None:
            return
        if self._place_robot:
            self.world._robot_x = gx
            self.world._robot_y = gy
            self._place_robot = False
            self._highlight_tool()
        else:
            self.drawing = True
            self.world._grid[gy, gx] = self.current_tool
        self._render()

    def _on_canvas_motion(self, event):
        if not self.drawing:
            return
        gx, gy = self._canvas_to_grid(event)
        if gx is None:
            return
        self.world._grid[gy, gx] = self.current_tool
        self._render()

    def _on_canvas_release(self, event):
        self.drawing = False

    # --- Rendering ---
    def _render(self):
        self.ax.clear()
        # Build RGB image
        img = np.zeros((self.world._height, self.world._width, 3), dtype=np.uint8)
        for y in range(self.world._height):
            for x in range(self.world._width):
                img[y, x] = VGA_PALETTE[self._grid_val(x, y)]

        self.ax.imshow(img, origin="upper", extent=[0, self.world._width, self.world._height, 0],
                       interpolation="nearest")

        # Draw grid lines
        for i in range(self.world._width + 1):
            self.ax.axvline(i, color="gray", linewidth=0.5, alpha=0.5)
        for i in range(self.world._height + 1):
            self.ax.axhline(i, color="gray", linewidth=0.5, alpha=0.5)

        # Draw robot
        rx = self.world._robot_x + 0.5
        ry = self.world._robot_y + 0.5
        # Body circle
        body = Circle((rx, ry), 0.25, color="red", zorder=5)
        self.ax.add_patch(body)
        # Direction arrow
        angle_rad = [math.pi / 2, 0, -math.pi / 2, math.pi][self.world._robot_dir]
        dx = 0.3 * math.cos(angle_rad)
        dy = -0.3 * math.sin(angle_rad)
        arrow = FancyArrow(rx, ry, dx, dy, width=0.08, head_width=0.15,
                           head_length=0.08, fc="darkred", ec="darkred", zorder=6)
        self.ax.add_patch(arrow)

        self.ax.set_xlim(0, self.world._width)
        self.ax.set_ylim(self.world._height, 0)
        self.ax.set_aspect("equal")
        self.ax.set_xticks(range(self.world._width))
        self.ax.set_yticks(range(self.world._height))
        self.ax.tick_params(labelsize=7)

        self.canvas.draw()
        self._update_inventory_label()

    def _grid_val(self, x, y):
        return int(self.world._grid[y, x])

    def _update_inventory_label(self):
        if self.world._robot_inventory is None:
            self.inventory_label.config(text="Inventory: empty")
        elif self.world._robot_inventory == CELL_KEY:
            self.inventory_label.config(text="Inventory: Key")
        else:
            self.inventory_label.config(text=f"Inventory: #{self.world._robot_inventory}")

    # --- Robot actions ---
    def _action_forward(self):
        self.world.do_action(0)
        self._render()

    def _action_backward(self):
        self.world.do_action(1)
        self._render()

    def _action_turn_left(self):
        self.world.do_action(2)
        self._render()

    def _action_turn_right(self):
        self.world.do_action(3)
        self._render()

    def _action_pickup(self):
        success = self.world.do_action(4)
        self._render()
        if not success:
            self.inventory_label.config(text="Inventory: (nothing to pick up)")

    def _action_use(self):
        success = self.world.do_action(5)
        self._render()
        if not success:
            self.inventory_label.config(text="Inventory: (can't use here)")

    def _action_drop(self):
        success = self.world.do_action(6)
        self._render()
        if not success:
            self.inventory_label.config(text="Inventory: (can't drop here)")

    def _print_observation(self):
        obs = self.world.get_observation()
        print("=" * 40)
        print("Observation:")
        print(f"  Pose: x={obs['pose'][0]}, y={obs['pose'][1]}, yaw={obs['pose'][2]}")
        print(f"  Inventory: {obs['inventory']}")
        print(f"  Grid ({self.world._height}x{self.world._width}):")
        print(obs["grid"])
        print("=" * 40)

    # --- File operations ---
    def _new_world(self):
        self.world.reset()
        self.current_file = None
        self.root.title("Grid World Editor")
        self._render()

    def _load_world(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            if self.world.load_png(filepath):
                self.current_file = filepath
                self.root.title(f"Grid World Editor - {filepath}")
                self._render()

    def _save_world(self):
        if self.current_file:
            self.world.save_png(self.current_file)
        else:
            self._save_world_as()

    def _save_world_as(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            self.world.save_png(filepath)
            self.current_file = filepath
            self.root.title(f"Grid World Editor - {filepath}")

    def _on_close(self):
        plt.close(self.fig)
        self.root.destroy()

    # --- Public API ---
    ACTIONS = GridWorld.ACTIONS

    def do_action(self, action_id):
        """Execute a robot action by integer index. Returns result of action."""
        result = self.world.do_action(action_id)
        self._render()
        return result

    def get_obs(self):
        """Return current observation dict: grid, pose, inventory."""
        return self.world.get_observation()

    def load(self, filepath):
        """Load world from PNG file."""
        if self.world.load_png(filepath):
            self.current_file = filepath
            self.root.title(f"Grid World Editor - {filepath}")
            self._render()
            return True
        return False

    def save(self):
        """Save world to current file. Returns False if no file set."""
        if self.current_file:
            self.world.save_png(self.current_file)
            return True
        return False

    def save_as(self, filepath):
        """Save world to specified PNG file."""
        self.world.save_png(filepath)
        self.current_file = filepath
        self.root.title(f"Grid World Editor - {filepath}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    editor = GridWorldEditor()
    editor.run()
