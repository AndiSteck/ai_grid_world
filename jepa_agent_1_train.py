"""
JEPA training on world1.png grid world.
Trains an encoder + predictor with EMA target encoder.
Saves model + metadata to jepa1/jepa_<timestamp>.pt
"""

import sys
import os
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, "/workspaces/python_play")
from grid_world_editor import GridWorld, DIR_EAST

# ----------------------------
# Config
# ----------------------------
WORLD_PATH = "/workspaces/python_play/worlds/world1.png"
OUTPUT_DIR = "/workspaces/python_play/jepa1"

EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_ACTIONS = 7
INPUT_DIM = 104  # 100 grid cells + x + y + dir + inventory

# Training params
NUM_TRAJECTORIES = 200
TRAJECTORY_LEN = 100
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
EMA_DECAY = 0.99


# ----------------------------
# Observation encoding
# ----------------------------
def encode_observation(obs):
    """Encode observation dict to flat tensor (104,)."""
    grid = obs["grid"].flatten().astype(np.float32) / 15.0  # normalize to [0, 1]
    x, y, d = obs["pose"]
    pose = np.array([x / 9.0, y / 9.0, d / 3.0], dtype=np.float32)
    inv = np.array([1.0 if obs["inventory"] is not None else 0.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([grid, pose, inv]))


def encode_action(action_id):
    """One-hot encode action (7,)."""
    v = torch.zeros(NUM_ACTIONS)
    v[action_id] = 1.0
    return v


# ----------------------------
# Model
# ----------------------------
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
# Data collection
# ----------------------------
def collect_trajectories(num_traj, traj_len):
    """Collect random trajectories as (obs, action, next_obs) tuples."""
    data = []
    for _ in range(num_traj):
        w = GridWorld()
        w.load_png(WORLD_PATH)
        # Random start pose on any valid cell (both sides of wall)
        while True:
            start_x = random.randint(0, 9)
            start_y = random.randint(0, 9)
            # Skip walls/doors - robot must start on walkable cell
            obs_check = w.get_observation()
            if obs_check["grid"][start_y, start_x] in (15, 2):  # empty or goal
                break
        start_dir = random.randint(0, 3)
        w.set_start_pose(start_x, start_y, start_dir)

        obs = w.get_observation()
        for _ in range(traj_len):
            action = random.randint(0, NUM_ACTIONS - 1)
            w.do_action(action)
            next_obs = w.get_observation()
            data.append((obs, action, next_obs))
            obs = next_obs
    return data


# ----------------------------
# Training
# ----------------------------
def train():
    print("=" * 60)
    print("JEPA Training on world1.png")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Input dim: {INPUT_DIM}")
    print(f"  Trajectories: {NUM_TRAJECTORIES} x {TRAJECTORY_LEN} steps")
    print(f"  Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LR}")
    print(f"  EMA decay: {EMA_DECAY}")
    print("=" * 60)

    # Collect data
    print("\nCollecting trajectories...")
    raw_data = collect_trajectories(NUM_TRAJECTORIES, TRAJECTORY_LEN)
    print(f"  Total transitions: {len(raw_data)}")

    # Pre-encode all data
    print("Encoding observations...")
    obs_tensors = torch.stack([encode_observation(d[0]) for d in raw_data])
    act_tensors = torch.stack([encode_action(d[1]) for d in raw_data])
    next_obs_tensors = torch.stack([encode_observation(d[2]) for d in raw_data])
    print(f"  obs shape: {obs_tensors.shape}")

    # Models
    encoder = Encoder()
    target_encoder = Encoder()
    predictor = Predictor()

    target_encoder.load_state_dict(encoder.state_dict())
    target_encoder.eval()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print("\nTraining...")
    n_samples = len(raw_data)
    losses = []
    train_start = time.time()

    for epoch in range(EPOCHS):
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            obs_b = obs_tensors[idx]
            act_b = act_tensors[idx]
            next_obs_b = next_obs_tensors[idx]

            # Forward
            z = encoder(obs_b)
            z_pred = predictor(z, act_b)

            with torch.no_grad():
                z_target = target_encoder(next_obs_b)

            # Loss: MSE in embedding space
            loss = ((z_pred - z_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # EMA update
        with torch.no_grad():
            for p, tp in zip(encoder.parameters(), target_encoder.parameters()):
                tp.data = EMA_DECAY * tp.data + (1 - EMA_DECAY) * p.data

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss = {avg_loss:.6f} | lr = {scheduler.get_last_lr()[0]:.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.1f}s")

    # ----------------------------
    # Quick sanity tests
    # ----------------------------
    print("\n" + "=" * 60)
    print("Sanity Tests")
    print("=" * 60)

    encoder.eval()
    target_encoder.eval()
    predictor.eval()

    # Test 1: Same observation should give similar embeddings
    w = GridWorld()
    w.load_png(WORLD_PATH)
    w.set_start_pose(0, 0, DIR_EAST)
    obs1 = w.get_observation()
    z1 = encoder(encode_observation(obs1).unsqueeze(0))
    z1b = encoder(encode_observation(obs1).unsqueeze(0))
    diff_same = ((z1 - z1b) ** 2).mean().item()
    print(f"  Same obs embedding diff: {diff_same:.8f} (should be ~0)")

    # Test 2: Different observations should give different embeddings
    w.do_action(0)  # move forward
    obs2 = w.get_observation()
    z2 = encoder(encode_observation(obs2).unsqueeze(0))
    diff_diff = ((z1 - z2) ** 2).mean().item()
    print(f"  Different obs embedding diff: {diff_diff:.6f} (should be > 0)")

    # Test 3: Predictor accuracy on known transition
    w2 = GridWorld()
    w2.load_png(WORLD_PATH)
    w2.set_start_pose(1, 1, DIR_EAST)
    obs_before = w2.get_observation()
    w2.do_action(0)  # forward
    obs_after = w2.get_observation()

    z_before = encoder(encode_observation(obs_before).unsqueeze(0))
    z_pred = predictor(z_before, encode_action(0).unsqueeze(0))
    z_actual = target_encoder(encode_observation(obs_after).unsqueeze(0))
    pred_error = ((z_pred - z_actual) ** 2).mean().item()
    print(f"  Predictor error (forward from 1,1): {pred_error:.6f}")

    # Test 4: Embedding norm (check it's not collapsed)
    norms = z1.norm(dim=-1).item()
    print(f"  Embedding L2 norm: {norms:.4f} (should be > 0, not collapsed)")

    # Test 5: Variance across batch
    sample_idx = torch.randperm(n_samples)[:100]
    z_batch = encoder(obs_tensors[sample_idx])
    z_var = z_batch.var(dim=0).mean().item()
    print(f"  Embedding variance (100 samples): {z_var:.6f} (should be > 0)")

    # ----------------------------
    # Save model + metadata
    # ----------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"jepa_{timestamp}.pt")

    metadata = {
        "timestamp": timestamp,
        "world": WORLD_PATH,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "input_dim": INPUT_DIM,
        "num_actions": NUM_ACTIONS,
        "num_trajectories": NUM_TRAJECTORIES,
        "trajectory_len": TRAJECTORY_LEN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "ema_decay": EMA_DECAY,
        "train_time_seconds": round(train_time, 2),
        "final_loss": losses[-1],
        "loss_history": losses,
        "tests": {
            "same_obs_diff": diff_same,
            "different_obs_diff": diff_diff,
            "predictor_error": pred_error,
            "embedding_norm": norms,
            "embedding_variance": z_var,
        },
    }

    torch.save({
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "metadata": metadata,
    }, save_path)

    print(f"\nModel saved to: {save_path}")
    print(f"Metadata: {json.dumps(metadata, indent=2, default=str)}")
    print("Done.")


if __name__ == "__main__":
    train()
