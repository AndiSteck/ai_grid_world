"""
JEPA Agent Runner - BFS Planning in Latent Space.
Plans a path to the goal cell using BFS over the JEPA predictor,
then executes step-by-step.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch
import torch.nn as nn
import math
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrow, Circle

import sys
sys.path.insert(0, "/workspaces/python_play")
from grid_world_editor import (
    GridWorld, VGA_PALETTE, CELL_WALL, CELL_EMPTY, CELL_GOAL, TOOL_NAMES,
    DIR_NORTH, DIR_EAST, DIR_SOUTH, DIR_WEST,
)


# ----------------------------
# Model definitions (must match training)
# ----------------------------
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_ACTIONS = 4  # forward, backward, turn_left, turn_right
INPUT_DIM = 103  # 100 grid cells + x + y + dir


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM + NUM_ACTIONS, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
        )

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)


# ----------------------------
# Encoding helpers
# ----------------------------
def encode_observation(obs):
    """Encode observation dict to flat tensor (103,)."""
    grid = obs["grid"].flatten().astype(np.float32) / 15.0
    x, y, d = obs["pose"]
    pose = np.array([x / 9.0, y / 9.0, d / 3.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([grid, pose]))


def encode_action(action_id):
    """One-hot encode action (4,)."""
    v = torch.zeros(NUM_ACTIONS)
    v[action_id] = 1.0
    return v


# ----------------------------
# Observation database
# ----------------------------
def build_observation_database(target_encoder, world):
    """Collect all reachable observations and their target embeddings."""
    obs_list = []

    if isinstance(world, str):
        w = GridWorld()
        w.load_png(world)
    else:
        w = world

    for sy in range(10):
        for sx in range(10):
            if w._grid[sy, sx] == CELL_WALL:
                continue
            for sd in range(4):
                w.set_start_pose(sx, sy, sd)
                obs_list.append(w.get_observation())

    encoded = torch.stack([encode_observation(o) for o in obs_list])
    target_encoder.eval()
    with torch.no_grad():
        z_all = target_encoder(encoded)

    return obs_list, z_all


def find_nearest_obs_index(z_pred, z_database):
    """Find index of nearest observation in database."""
    z_pred = z_pred.reshape(1, -1)
    dists = ((z_database - z_pred) ** 2).sum(dim=-1)
    return dists.argmin().item()


# ----------------------------
# BFS Planner in Latent Space
# ----------------------------
def bfs_plan(encoder, predictor, z_database, obs_list, start_obs, max_depth=30):
    """BFS in latent space to find action sequence reaching a goal cell.

    Uses the predictor to simulate transitions and snaps predicted embeddings
    to the nearest known state in the database for visited-state tracking.

    Returns: list of action_ids, or None if no path found.
    """
    # Identify goal state indices in database
    goal_indices = set()
    for i, obs in enumerate(obs_list):
        px, py, _ = obs["pose"]
        if obs["grid"][py, px] == CELL_GOAL:
            goal_indices.add(i)

    if not goal_indices:
        return None, "No goal cell in world"

    # Encode start state
    with torch.no_grad():
        z_start = encoder(encode_observation(start_obs).unsqueeze(0))

    start_idx = find_nearest_obs_index(z_start, z_database)

    if start_idx in goal_indices:
        return [], "Already at goal"

    # BFS: queue of (state_index, z_embedding, action_sequence)
    visited = {start_idx}
    queue = deque([(start_idx, z_start, [])])

    # Precompute action encodings
    action_encodings = [encode_action(a).unsqueeze(0) for a in range(NUM_ACTIONS)]

    with torch.no_grad():
        while queue:
            state_idx, z_current, actions = queue.popleft()

            if len(actions) >= max_depth:
                continue

            for action_id in range(NUM_ACTIONS):
                z_next = predictor(z_current, action_encodings[action_id])
                next_idx = find_nearest_obs_index(z_next, z_database)

                if next_idx in visited:
                    continue

                new_actions = actions + [action_id]

                if next_idx in goal_indices:
                    return new_actions, f"Found path: {len(new_actions)} steps"

                visited.add(next_idx)
                queue.append((next_idx, z_next, new_actions))

    return None, f"No path found (explored {len(visited)} states, max_depth={max_depth})"


# ----------------------------
# GUI
# ----------------------------
class JEPARunner:
    def __init__(self):
        self.world = GridWorld()
        self.current_tool = CELL_WALL
        self.current_file = None
        self.drawing = False

        # JEPA model state
        self.encoder = None
        self.target_encoder = None
        self.predictor = None
        self.obs_database = None
        self.z_database = None
        self.world_dirty = False

        # Plan state
        self.plan = None  # list of action_ids
        self.plan_path = None  # list of (x, y, dir) for visualization

        # Main window
        self.root = tk.Tk()
        self.root.title("JEPA Agent Runner")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Menu bar
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load World...", command=self._load_world)
        filemenu.add_command(label="Save World", command=self._save_world)
        filemenu.add_command(label="Save World As...", command=self._save_world_as)
        filemenu.add_separator()
        filemenu.add_command(label="Load JEPA Model...", command=self._load_model)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # Main layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel
        tool_frame = tk.Frame(main_frame, width=192, padx=5, pady=5)
        tool_frame.pack(side=tk.LEFT, fill=tk.Y)
        tool_frame.pack_propagate(False)

        tk.Label(tool_frame, text="Draw Tools", font=("Arial", 10, "bold")).pack(pady=(5, 5))

        self.tool_buttons = {}
        for cell_type, name in TOOL_NAMES.items():
            color = "#{:02x}{:02x}{:02x}".format(*VGA_PALETTE[cell_type])
            fg = "white" if cell_type in (CELL_WALL,) else "black"
            btn = tk.Button(
                tool_frame, text=name, width=12,
                bg=color, fg=fg, relief=tk.RAISED,
                command=lambda ct=cell_type: self._select_tool(ct)
            )
            btn.pack(pady=1)
            self.tool_buttons[cell_type] = btn

        tk.Button(tool_frame, text="Place Robot", width=12,
                  command=self._place_robot_mode).pack(pady=2)

        tk.Label(tool_frame, text="", height=0).pack()
        tk.Label(tool_frame, text="Actions", font=("Arial", 10, "bold")).pack(pady=(3, 3))

        self.action_buttons = []
        for label, aid in [("Forward", 0), ("Backward", 1), ("Turn Left", 2), ("Turn Right", 3)]:
            btn = tk.Button(tool_frame, text=label, width=12,
                            command=lambda a=aid: self._do_action(a))
            btn.pack(pady=1)
            self.action_buttons.append(btn)

        # Planning section
        tk.Label(tool_frame, text="", height=0).pack()
        tk.Label(tool_frame, text="Planning", font=("Arial", 10, "bold")).pack(pady=(3, 3))

        self.plan_btn = tk.Button(tool_frame, text="Plan BFS", width=12,
                                  command=self._plan_bfs, state=tk.DISABLED)
        self.plan_btn.pack(pady=2)

        self.step_btn = tk.Button(tool_frame, text="Step", width=12,
                                  command=self._step_plan, state=tk.DISABLED)
        self.step_btn.pack(pady=2)

        self.plan_label = tk.Label(tool_frame, text="Plan: --", font=("Arial", 9, "bold"))
        self.plan_label.pack(pady=1)

        tk.Label(tool_frame, text="", height=0).pack()
        self.rebuild_btn = tk.Button(tool_frame, text="Rebuild Obs DB", width=12,
                                     command=self._rebuild_obs_database, state=tk.DISABLED)
        self.rebuild_btn.pack(pady=3)

        self.model_label = tk.Label(tool_frame, text="Model: none", font=("Arial", 8),
                                     wraplength=180)
        self.model_label.pack(pady=2)

        # Canvas area - single plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.04)

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Mouse events
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
        if event.inaxes != self.ax:
            return None, None
        gx = int(event.xdata)
        gy = int(event.ydata)
        if 0 <= gx < 10 and 0 <= gy < 10:
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
            self._mark_world_dirty()
        self._clear_plan()
        self._render()

    def _on_canvas_motion(self, event):
        if not self.drawing:
            return
        gx, gy = self._canvas_to_grid(event)
        if gx is None:
            return
        self.world._grid[gy, gx] = self.current_tool
        self._mark_world_dirty()
        self._render()

    def _on_canvas_release(self, event):
        self.drawing = False

    # --- Actions ---
    def _do_action(self, action_id):
        """Perform action on real world (manual control)."""
        self.world.do_action(action_id)
        self._clear_plan()
        self._render()

    # --- Planning ---
    def _plan_bfs(self):
        """Run BFS in latent space from current observation."""
        if self.encoder is None or self.predictor is None or self.obs_database is None:
            return

        obs = self.world.get_observation()
        self.plan_label.config(text="Planning...", fg="blue")
        self.root.update()

        plan, msg = bfs_plan(
            self.encoder, self.predictor, self.z_database,
            self.obs_database, obs
        )

        if plan is None:
            self.plan = None
            self.plan_path = None
            self.plan_label.config(text=f"Plan: FAILED", fg="red")
            messagebox.showwarning("BFS", msg)
        elif len(plan) == 0:
            self.plan = []
            self.plan_path = []
            self.plan_label.config(text="Plan: at goal!", fg="green")
        else:
            self.plan = plan
            self._compute_plan_path(obs)
            self.plan_label.config(text=f"Plan: {len(plan)} steps", fg="green")
            self.step_btn.config(state=tk.NORMAL)

        self._render()

    def _compute_plan_path(self, start_obs):
        """Compute predicted path poses for visualization."""
        path = [(start_obs["pose"][0], start_obs["pose"][1], start_obs["pose"][2])]

        with torch.no_grad():
            z = self.encoder(encode_observation(start_obs).unsqueeze(0))
            for action_id in self.plan:
                a_enc = encode_action(action_id).unsqueeze(0)
                z = self.predictor(z, a_enc)
                idx = find_nearest_obs_index(z, self.z_database)
                obs = self.obs_database[idx]
                px, py, pd = obs["pose"]
                path.append((px, py, pd))

        self.plan_path = path

    def _step_plan(self):
        """Execute one step of the plan."""
        if not self.plan:
            self.step_btn.config(state=tk.DISABLED)
            return

        action_id = self.plan.pop(0)
        self.world.do_action(action_id)

        # Remove first entry from path (the position we just left)
        if self.plan_path:
            self.plan_path.pop(0)

        if not self.plan:
            self.step_btn.config(state=tk.DISABLED)
            if self.world.on_goal():
                self.plan_label.config(text="Plan: REACHED GOAL!", fg="green")
            else:
                self.plan_label.config(text="Plan: done (0 left)", fg="black")
        else:
            self.plan_label.config(text=f"Plan: {len(self.plan)} steps left", fg="blue")

        self._render()

    def _clear_plan(self):
        """Clear current plan."""
        self.plan = None
        self.plan_path = None
        self.step_btn.config(state=tk.DISABLED)
        self.plan_label.config(text="Plan: --", fg="black")

    def _mark_world_dirty(self):
        if not self.world_dirty:
            self.world_dirty = True
            self.plan_btn.config(state=tk.DISABLED)
            for btn in self.action_buttons:
                btn.config(state=tk.DISABLED)
            if self.target_encoder is not None:
                self.rebuild_btn.config(state=tk.NORMAL)

    def _rebuild_obs_database(self):
        if self.target_encoder is None:
            return
        obs = self.world.get_observation()
        rx, ry, rd = obs["pose"]
        self.obs_database, self.z_database = build_observation_database(
            self.target_encoder, self.world)
        self.world.set_start_pose(rx, ry, rd)
        self.world_dirty = False
        self.rebuild_btn.config(state=tk.DISABLED)
        self.plan_btn.config(state=tk.NORMAL)
        for btn in self.action_buttons:
            btn.config(state=tk.NORMAL)
        self.model_label.config(
            text=f"Model: {len(self.obs_database)} states (rebuilt)")

    # --- Rendering ---
    def _render(self):
        self.ax.clear()
        obs = self.world.get_observation()
        grid = obs["grid"]
        px, py, pd = obs["pose"]

        # Build RGB image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        for y in range(10):
            for x in range(10):
                val = int(np.clip(grid[y, x], 0, 15))
                img[y, x] = VGA_PALETTE[val]

        self.ax.imshow(img, origin="upper", extent=[0, 10, 10, 0], interpolation="nearest")

        # Grid lines
        for i in range(11):
            self.ax.axvline(i, color="gray", linewidth=0.5, alpha=0.5)
            self.ax.axhline(i, color="gray", linewidth=0.5, alpha=0.5)

        # Robot
        rx = px + 0.5
        ry = py + 0.5
        body = Circle((rx, ry), 0.25, color="red", zorder=5)
        self.ax.add_patch(body)
        angle_rad = [math.pi / 2, 0, -math.pi / 2, math.pi][pd]
        dx = 0.3 * math.cos(angle_rad)
        dy = -0.3 * math.sin(angle_rad)
        arrow = FancyArrow(rx, ry, dx, dy, width=0.08, head_width=0.15,
                           head_length=0.08, fc="darkred", ec="darkred", zorder=6)
        self.ax.add_patch(arrow)

        # Draw plan path
        if self.plan_path and len(self.plan_path) > 1:
            for i in range(len(self.plan_path) - 1):
                x1, y1, _ = self.plan_path[i]
                x2, y2, d2 = self.plan_path[i + 1]
                # Line segment
                self.ax.plot(
                    [x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5],
                    color="cyan", linewidth=2, alpha=0.7, zorder=3
                )
                # Dot at each waypoint
                self.ax.plot(x2 + 0.5, y2 + 0.5, 'o', color="cyan",
                            markersize=5, alpha=0.8, zorder=4)

            # Mark goal end with a larger marker
            gx, gy, gd = self.plan_path[-1]
            self.ax.plot(gx + 0.5, gy + 0.5, '*', color="yellow",
                        markersize=14, markeredgecolor="black", zorder=7)

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(10, 0)
        self.ax.set_aspect("equal")
        self.ax.set_xticks(range(10))
        self.ax.set_yticks(range(10))
        self.ax.tick_params(labelsize=6)
        self.ax.set_title("JEPA Agent Runner", fontsize=10)

        self.canvas.draw()

    # --- File operations ---
    def _load_world(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            if self.world.load_png(filepath):
                self.current_file = filepath
                self._clear_plan()
                self.root.title(f"JEPA Agent Runner - {filepath}")
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
            self.root.title(f"JEPA Agent Runner - {filepath}")

    def _load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

            self.encoder = Encoder()
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.encoder.eval()

            self.target_encoder = Encoder()
            self.target_encoder.load_state_dict(checkpoint["target_encoder"])
            self.target_encoder.eval()

            self.predictor = Predictor()
            self.predictor.load_state_dict(checkpoint["predictor"])
            self.predictor.eval()

            self._clear_plan()

            # Build observation database
            metadata = checkpoint.get("metadata", {})
            world_path = metadata.get("world", self.current_file)
            if world_path == "random":
                world_path = self.current_file
            if world_path:
                self.model_label.config(text="Model: building obs database...")
                self.root.update()
                self.obs_database, self.z_database = build_observation_database(
                    self.target_encoder, world_path)
                self.model_label.config(
                    text=f"Model: {filepath.split('/')[-1]} ({len(self.obs_database)} states)")
                self.plan_btn.config(state=tk.NORMAL)
            else:
                self.obs_database = None
                self.z_database = None
                self.model_label.config(text=f"Model: {filepath.split('/')[-1]} (no world)")

            self._render()
            messagebox.showinfo("Model Loaded",
                f"Loaded: {filepath}\nStates: {len(self.obs_database) if self.obs_database else 0}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def _on_close(self):
        plt.close(self.fig)
        self.root.destroy()


if __name__ == "__main__":
    app = JEPARunner()
    app.run()
