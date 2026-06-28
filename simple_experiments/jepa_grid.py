import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# 1. Tiny Grid World
# ----------------------------
class TinyGridWorld:
    """
    3x3 grid:
    R = robot
    K = key
    G = goal
    X = wall
    . = empty
    """

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(self):
        # Static terrain (never changes)
        self.terrain = [
            [".", ".", "K"],
            [".", "X", "."],
            [".", ".", "G"]
        ]
        self.reset()

    def reset(self):
        self.robot_pos = (0, 0)
        return self._get_obs()

    def _get_obs(self):
        obs = [row[:] for row in self.terrain]
        rx, ry = self.robot_pos
        obs[rx][ry] = "R"
        return np.array(obs)

    def step(self, action):
        x, y = self.robot_pos

        if action == 0: # UP
            nx, ny = x-1, y
        elif action == 1: # DOWN
            nx, ny = x+1, y
        elif action == 2: # LEFT
            nx, ny = x, y-1
        else: # RIGHT
            nx, ny = x, y+1

        # bounds check
        if nx < 0 or ny < 0 or nx >= 3 or ny >= 3:
            nx, ny = x, y

        # wall check
        if self.terrain[nx][ny] == "X":
            nx, ny = x, y

        self.robot_pos = (nx, ny)
        return self._get_obs()


# ----------------------------
# 2. Encoding
# ----------------------------
VOCAB = {
    "R": [1,0,0,0,0],
    "X": [0,1,0,0,0],
    "K": [0,0,1,0,0],
    "G": [0,0,0,1,0],
    ".": [0,0,0,0,1],
}

def encode(obs):
    out = []
    for row in obs:
        for cell in row:
            out.extend(VOCAB[cell])
    return torch.tensor(out, dtype=torch.float32)


def action_one_hot(a):
    v = torch.zeros(4)
    v[a] = 1.0
    return v


# ----------------------------
# 3. JEPA Model
# ----------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)


# ----------------------------
# 4. Data collection
# ----------------------------
def generate_data(env, steps=300):
    data = []
    obs = env.reset()

    for _ in range(steps):
        action = random.randint(0, 3)
        next_obs = env.step(action)
        data.append((obs, action, next_obs))
        obs = next_obs

    return data

# ----------------------------
# 5. Training
# ----------------------------
def train():
    env = TinyGridWorld()

    encoder = Encoder()
    target_encoder = Encoder()
    predictor = Predictor()

    target_encoder.load_state_dict(encoder.state_dict())
    target_encoder.eval()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3
    )

    data = generate_data(env)

    for epoch in range(50):
        total_loss = 0

        for obs, action, next_obs in data:
            obs_t = encode(obs)
            next_obs_t = encode(next_obs)
            a_t = action_one_hot(action)

            z = encoder(obs_t)
            z_pred = predictor(z, a_t)

            with torch.no_grad():
                z_true = target_encoder(next_obs_t)

            loss = ((z_pred - z_true) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # EMA update (JEPA trick)
        with torch.no_grad():
            for p, tp in zip(encoder.parameters(), target_encoder.parameters()):
                tp.data = 0.99 * tp.data + 0.01 * p.data

        print(f"Epoch {epoch:02d} | loss = {total_loss:.4f}")

    return encoder, target_encoder, predictor


# ----------------------------
# 5. Training optimized with batching and scheduler
# ----------------------------
def train_optimized():
    env = TinyGridWorld()

    encoder = Encoder()
    target_encoder = Encoder()
    predictor = Predictor()

    target_encoder.load_state_dict(encoder.state_dict())
    target_encoder.eval()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    data = generate_data(env, steps=1000)
    batch_size = 32

    for epoch in range(100):
        random.shuffle(data)
        total_loss = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            obs_batch = torch.stack([encode(o) for o, _, _ in batch])
            next_batch = torch.stack([encode(n) for _, _, n in batch])
            act_batch = torch.stack([action_one_hot(a) for _, a, _ in batch])

            z = encoder(obs_batch)
            z_pred = predictor(z, act_batch)

            with torch.no_grad():
                z_true = target_encoder(next_batch)

            loss = ((z_pred - z_true) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # EMA update (JEPA trick) - slower for stable targets
        with torch.no_grad():
            for p, tp in zip(encoder.parameters(), target_encoder.parameters()):
                tp.data = 0.996 * tp.data + 0.004 * p.data

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | loss = {total_loss:.4f}")

    return encoder, target_encoder, predictor


# ----------------------------
# 6. Prediction demo
# ----------------------------
def decode_obs(encoded_obs):
    """Decode a 45-dim encoded observation back to a 3x3 grid."""
    inv_vocab = {tuple(v): k for k, v in VOCAB.items()}
    cells = encoded_obs.reshape(9, 5)
    grid = []
    for cell in cells:
        # Find closest vocab entry
        best_key = min(inv_vocab.keys(), key=lambda k: ((torch.tensor(k, dtype=torch.float32) - cell) ** 2).sum())
        grid.append(inv_vocab[best_key])
    return np.array(grid).reshape(3, 3)


def find_nearest_obs(z_pred, target_encoder, all_obs_encodings):
    """Find the observation whose target encoding is closest to z_pred."""
    best_dist = float("inf")
    best_obs = None
    for obs_t, obs_raw in all_obs_encodings:
        with torch.no_grad():
            z = target_encoder(obs_t)
        dist = ((z_pred - z) ** 2).sum().item()
        if dist < best_dist:
            best_dist = dist
            best_obs = obs_raw
    return best_obs, best_dist


def demo_prediction(encoder, target_encoder, predictor):
    """Demonstrate that the model can predict next_obs given obs + action."""
    env = TinyGridWorld()

    # Collect all reachable states for nearest-neighbor decoding
    all_obs = []
    obs = env.reset()
    all_obs.append(obs.copy())
    for _ in range(200):
        action = random.randint(0, 3)
        obs = env.step(action)
        # Only add unique observations
        if not any(np.array_equal(obs, o) for o in all_obs):
            all_obs.append(obs.copy())

    all_obs_encodings = [(encode(o), o) for o in all_obs]

    # Now run a few steps and show prediction vs actual
    env.reset()
    obs = env.reset()
    print("\n" + "=" * 50)
    print("PREDICTION DEMO: obs + action -> predicted next_obs")
    print("=" * 50)

    for step in range(5):
        action = random.randint(0, 3)
        action_name = TinyGridWorld.ACTIONS[action]

        # Model prediction
        obs_t = encode(obs)
        a_t = action_one_hot(action)
        with torch.no_grad():
            z = encoder(obs_t)
            z_pred = predictor(z, a_t)

        # Find nearest known observation to predicted embedding
        predicted_obs, dist = find_nearest_obs(z_pred, target_encoder, all_obs_encodings)

        # Actual next obs
        actual_next_obs = env.step(action)

        match = "CORRECT" if np.array_equal(predicted_obs, actual_next_obs) else "WRONG"

        def fmt_grid(g):
            rows = []
            for r in range(3):
                rows.append("  " + " ".join(g[r]))
            return "\n".join(rows)

        print(f"\nStep {step + 1} | Action: {action_name} | {match} (dist={dist:.4f})")
        print(f"  Current obs:")
        print(fmt_grid(obs))
        print(f"  Predicted next:")
        print(fmt_grid(predicted_obs))
        print(f"  Actual next:")
        print(fmt_grid(actual_next_obs))

        obs = actual_next_obs


# ----------------------------
# 7. Goal-directed planning demo
# ----------------------------
def demo_planning(encoder, predictor):
    """Use JEPA to plan actions toward a goal in latent space (multi-step lookahead)."""
    env = TinyGridWorld()
    obs = env.reset()

    # Goal: robot at (2,2) — the G cell
    goal_grid = np.array([
        [".", ".", "K"],
        [".", "X", "."],
        [".", ".", "R"]
    ])
    with torch.no_grad():
        z_goal = encoder(encode(goal_grid))

    print("\n" + "=" * 50)
    print("PLANNING DEMO: multi-step lookahead toward goal")
    print("=" * 50)

    def fmt_grid(g):
        rows = []
        for r in range(3):
            rows.append("  " + " ".join(g[r]))
        return "\n".join(rows)

    print(f"  Goal state:")
    print(fmt_grid(goal_grid))
    print()

    def plan_best_action(z_start, depth=6, blocked_actions=None):
        """Tree search: expand all action sequences up to depth, return best first action."""
        if blocked_actions is None:
            blocked_actions = set()

        best_action = 0
        best_dist = float("inf")
        available = [a for a in range(4) if a not in blocked_actions]
        if not available:
            available = list(range(4))

        with torch.no_grad():
            # Start with available first actions
            frontier = []
            for a in available:
                z_pred = predictor(z_start.unsqueeze(0), action_one_hot(a).unsqueeze(0)).squeeze(0)
                dist = ((z_pred - z_goal) ** 2).sum().item()
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
                frontier.append((z_pred, a))

            # Expand tree for remaining depth
            for _ in range(depth - 1):
                next_frontier = []
                for z, first_a in frontier:
                    for a in range(4):
                        z_pred = predictor(z.unsqueeze(0), action_one_hot(a).unsqueeze(0)).squeeze(0)
                        dist = ((z_pred - z_goal) ** 2).sum().item()
                        if dist < best_dist:
                            best_dist = dist
                            best_action = first_a
                        next_frontier.append((z_pred, first_a))
                # Keep top-k to avoid exponential blowup
                next_frontier.sort(key=lambda x: ((x[0] - z_goal) ** 2).sum().item())
                frontier = next_frontier[:32]

        return best_action, best_dist

    prev_pos = None
    stuck_count = 0
    blocked_actions = set()

    for step in range(15):
        obs_t = encode(obs)
        with torch.no_grad():
            z = encoder(obs_t)

        # If stuck (didn't move), block the last action tried
        if prev_pos == env.robot_pos:
            stuck_count += 1
            blocked_actions.add(last_action)
        else:
            stuck_count = 0
            blocked_actions = set()

        prev_pos = env.robot_pos
        best_action, best_dist = plan_best_action(z, depth=6, blocked_actions=blocked_actions)
        last_action = best_action

        action_name = TinyGridWorld.ACTIONS[best_action]
        next_obs = env.step(best_action)

        print(f"  Step {step+1}: chose {action_name:5s} (dist_to_goal={best_dist:.4f})")
        print(fmt_grid(next_obs))
        print()

        # Check if we reached the goal (robot at G position)
        if env.robot_pos == (2, 2):
            print("  >>> Reached the goal! <<<")
            break

        obs = next_obs


# ----------------------------
# 8. Run
# ----------------------------
if __name__ == "__main__":
    encoder, target_encoder, predictor = train()
    demo_prediction(encoder, target_encoder, predictor)
    demo_planning(encoder, predictor)
