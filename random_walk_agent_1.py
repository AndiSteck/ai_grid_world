"""
Random walk agent for world1.png.
Runs N episodes with random actions and reports statistics.
"""

import sys
import random
import numpy as np

sys.path.insert(0, "/workspaces/python_play")
from grid_world_editor import GridWorld, CELL_GOAL, DIR_EAST


def run_episode(max_steps=200000):
    w = GridWorld()
    w.load_png("/workspaces/python_play/worlds/world1.png")
    w.robot_x, w.robot_y, w.robot_dir = 0, 0, DIR_EAST

    for step in range(max_steps):
        action = random.randint(0, 6)
        if action == 0:
            w.move_forward()
        elif action == 1:
            w.move_backward()
        elif action == 2:
            w.turn_left()
        elif action == 3:
            w.turn_right()
        elif action == 4:
            w.pickup()
        elif action == 5:
            w.use_object()
        elif action == 6:
            w.drop_object()

        if w.grid[w.robot_y, w.robot_x] == CELL_GOAL:
            return step + 1
    return -1


def main():
    N = 50
    max_steps = 500000

    print(f"Random walk agent on world1.png")
    print(f"Trials: {N}, max steps per trial: {max_steps}")
    print("-" * 50)

    successes = []
    failures = 0

    for i in range(N):
        result = run_episode(max_steps=max_steps)
        if result > 0:
            successes.append(result)
            print(f"  Trial {i:3d}: solved in {result} steps")
        else:
            failures += 1
            print(f"  Trial {i:3d}: FAILED ({max_steps} steps)")

    print("-" * 50)
    print(f"Successes: {len(successes)}/{N}  | Success rate: {100*len(successes)/N:.1f}%")

    if successes:
        arr = np.array(successes)
        print(f"Mean:   {arr.mean():.0f} steps")
        print(f"Median: {np.median(arr):.0f} steps")
        print(f"Best:   {arr.min()} steps")
        print(f"Worst:  {arr.max()} steps")
        print(f"Std:    {arr.std():.0f} steps")
    else:
        print("No successes within step limit.")


if __name__ == "__main__":
    main()
