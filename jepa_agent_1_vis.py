"""
JEPA Visualization Tool for world1.png.
Shows real world + JEPA predicted world side by side.
Allows loading world/model, performing actions, and comparing predictions.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrow, Circle

import sys
sys.path.insert(0, "/workspaces/python_play")
from grid_world_editor import (
    GridWorld, VGA_PALETTE, CELL_WALL, CELL_EMPTY, TOOL_NAMES,
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
# Encoding / Decoding helpers
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
# Nearest-neighbor observation lookup
# ----------------------------
def build_observation_database(target_encoder, world):
    """Collect all reachable observations and their target embeddings.

    Enumerates all valid (position, direction) combinations in the world.
    Accepts a GridWorld instance or a file path string.
    """
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

    # Encode all observations with target encoder
    encoded = torch.stack([encode_observation(o) for o in obs_list])
    target_encoder.eval()
    with torch.no_grad():
        z_all = target_encoder(encoded)

    return obs_list, z_all


def find_nearest_obs(z_pred, obs_list, z_database):
    """Find the observation whose target encoding is closest to z_pred."""
    # z_pred: (1, embed_dim) or (embed_dim,)
    z_pred = z_pred.reshape(1, -1)
    dists = ((z_database - z_pred) ** 2).sum(dim=-1)
    best_idx = dists.argmin().item()
    best_dist = dists[best_idx].item()
    return obs_list[best_idx], best_dist


# ----------------------------
# Visualization GUI
# ----------------------------
class JEPAVisualizer:
    def __init__(self):
        self.world = GridWorld()
        self.current_tool = CELL_WALL
        self.current_file = None
        self.model_file = None
        self.drawing = False

        # JEPA model state
        self.encoder = None
        self.target_encoder = None
        self.predictor = None
        self.obs_database = None  # list of observations
        self.z_database = None    # target encoder embeddings
        self.predicted_obs = None
        self.last_match_score = None
        self.world_dirty = False  # True when world was modified since last obs DB build

        # Main window
        self.root = tk.Tk()
        self.root.title("JEPA Visualizer")
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

        # Left panel - tools and actions
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

        # Status section
        tk.Label(tool_frame, text="", height=0).pack()
        self.rebuild_btn = tk.Button(tool_frame, text="Rebuild Obs DB", width=12,
                                     command=self._rebuild_obs_database, state=tk.DISABLED)
        self.rebuild_btn.pack(pady=3)

        self.match_label = tk.Label(tool_frame, text="Match: --", font=("Arial", 9, "bold"))
        self.match_label.pack(pady=1)

        self.model_label = tk.Label(tool_frame, text="Model: none", font=("Arial", 8),
                                     wraplength=180)
        self.model_label.pack(pady=2)

        tk.Button(tool_frame, text="Reset Prediction", width=12,
                  command=self._reset_prediction).pack(pady=3)

        # Canvas area - two plots side by side
        self.fig, (self.ax_real, self.ax_pred) = plt.subplots(1, 2, figsize=(11, 5.5))
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.1)

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
        # Only handle clicks on the real world (left) axes
        if event.inaxes != self.ax_real:
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
        """Perform action, update JEPA prediction, compare."""
        # Get observation before action for JEPA prediction
        obs_before = self.world.get_observation()

        # Perform action on real world
        self.world.do_action(action_id)

        # Get actual observation after action
        obs_after = self.world.get_observation()

        # JEPA prediction
        if self.encoder is not None and self.predictor is not None and self.obs_database is not None:
            with torch.no_grad():
                # Always encode from real observation (single-step prediction)
                obs_enc = encode_observation(obs_before).unsqueeze(0)
                z_current = self.encoder(obs_enc)

                # Predict next embedding
                a_enc = encode_action(action_id).unsqueeze(0)
                z_pred = self.predictor(z_current, a_enc)

                # Find nearest known observation in target encoder space
                self.predicted_obs, best_dist = find_nearest_obs(
                    z_pred, self.obs_database, self.z_database)

                # Compute match: compare predicted embedding with actual
                obs_actual_enc = encode_observation(obs_after).unsqueeze(0)
                z_actual = self.target_encoder(obs_actual_enc)
                mse = ((z_pred - z_actual) ** 2).mean().item()
                self.last_match_score = mse

        self._render()

    def _reset_prediction(self):
        """Reset prediction state."""
        self.predicted_obs = None
        self.last_match_score = None
        self._render()

    def _mark_world_dirty(self):
        """Mark world as modified, disable actions until obs DB is rebuilt."""
        if not self.world_dirty:
            self.world_dirty = True
            for btn in self.action_buttons:
                btn.config(state=tk.DISABLED)
            if self.target_encoder is not None:
                self.rebuild_btn.config(state=tk.NORMAL)

    def _rebuild_obs_database(self):
        """Rebuild observation database from current world state."""
        if self.target_encoder is None:
            return
        # Save/restore robot pose since build enumerates all poses
        obs = self.world.get_observation()
        rx, ry, rd = obs["pose"]
        self.obs_database, self.z_database = build_observation_database(
            self.target_encoder, self.world)
        self.world.set_start_pose(rx, ry, rd)
        self.world_dirty = False
        self.rebuild_btn.config(state=tk.DISABLED)
        for btn in self.action_buttons:
            btn.config(state=tk.NORMAL)
        self.model_label.config(
            text=f"Model: {len(self.obs_database)} states (rebuilt)")

    # --- Rendering ---
    def _render(self):
        self.ax_real.clear()
        self.ax_pred.clear()

        # --- Real world ---
        self._render_world(self.ax_real, self.world.get_observation(), "Real World")

        # --- Predicted world ---
        if self.predicted_obs is not None:
            self._render_world(self.ax_pred, self.predicted_obs, "JEPA Prediction")
        else:
            self.ax_pred.set_title("JEPA Prediction", fontsize=10)
            self.ax_pred.text(5, 5, "No prediction yet\n(perform an action)",
                            ha='center', va='center', fontsize=11, color='gray')
            self.ax_pred.set_xlim(0, 10)
            self.ax_pred.set_ylim(10, 0)

        self.canvas.draw()
        self._update_status()

    def _render_world(self, ax, obs, title):
        """Render a world observation on given axes."""
        grid = obs["grid"]
        px, py, pd = obs["pose"]

        # Build RGB image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        for y in range(10):
            for x in range(10):
                val = int(np.clip(grid[y, x], 0, 15))
                img[y, x] = VGA_PALETTE[val]

        ax.imshow(img, origin="upper", extent=[0, 10, 10, 0], interpolation="nearest")

        # Grid lines
        for i in range(11):
            ax.axvline(i, color="gray", linewidth=0.5, alpha=0.5)
            ax.axhline(i, color="gray", linewidth=0.5, alpha=0.5)

        # Robot
        rx = px + 0.5
        ry = py + 0.5
        body = Circle((rx, ry), 0.25, color="red", zorder=5)
        ax.add_patch(body)
        angle_rad = [math.pi / 2, 0, -math.pi / 2, math.pi][pd]
        dx = 0.3 * math.cos(angle_rad)
        dy = -0.3 * math.sin(angle_rad)
        arrow = FancyArrow(rx, ry, dx, dy, width=0.08, head_width=0.15,
                           head_length=0.08, fc="darkred", ec="darkred", zorder=6)
        ax.add_patch(arrow)

        ax.set_xlim(0, 10)
        ax.set_ylim(10, 0)
        ax.set_aspect("equal")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=10)

    def _update_status(self):
        obs = self.world.get_observation()

        # Match indicator - compare predicted obs with actual obs
        if self.last_match_score is None or self.predicted_obs is None:
            self.match_label.config(text="Match: --", fg="black")
        else:
            pose_match = obs["pose"] == self.predicted_obs["pose"]
            if pose_match:
                self.match_label.config(text="Match: CORRECT", fg="green")
            else:
                self.match_label.config(text="Match: WRONG", fg="red")

    # --- File operations ---
    def _load_world(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            if self.world.load_png(filepath):
                self.current_file = filepath
                self.predicted_embedding = None
                self.last_match_score = None
                self.root.title(f"JEPA Visualizer - {filepath}")
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
            self.root.title(f"JEPA Visualizer - {filepath}")

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

            self.model_file = filepath
            self.predicted_obs = None
            self.last_match_score = None

            # Build observation database for nearest-neighbor lookup
            metadata = checkpoint.get("metadata", {})
            world_path = metadata.get("world", self.current_file)
            # "random" means model was trained on random worlds, not a specific file
            if world_path == "random":
                world_path = self.current_file
            if world_path:
                self.model_label.config(text=f"Model: building obs database...")
                self.root.update()
                self.obs_database, self.z_database = build_observation_database(
                    self.target_encoder, world_path)
                self.model_label.config(
                    text=f"Model: {filepath.split('/')[-1]} ({len(self.obs_database)} states)")
            else:
                self.obs_database = None
                self.z_database = None
                self.model_label.config(text=f"Model: {filepath.split('/')[-1]} (no world)")

            self._render()
            messagebox.showinfo("Model Loaded",
                f"Loaded: {filepath}\nObservation database: {len(self.obs_database) if self.obs_database else 0} states")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def _on_close(self):
        plt.close(self.fig)
        self.root.destroy()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    vis = JEPAVisualizer()
    vis.run()
