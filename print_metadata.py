#!/usr/bin/env python3
"""Print metadata from a JEPA model checkpoint."""

import sys
import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: python print_metadata.py <model.pt>")
        sys.exit(1)

    filepath = sys.argv[1]
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})

    print(f"\n{'=' * 60}")
    print(f"Model: {filepath}")
    print(f"{'=' * 60}")
    for k, v in metadata.items():
        if k == "loss_history":
            print(f"  {k}: [{v[0]:.6f} ... {v[-1]:.6f}] ({len(v)} epochs)")
        elif k == "tests":
            print(f"  {k}:")
            for tk_, tv in v.items():
                print(f"    {tk_}: {tv}")
        else:
            print(f"  {k}: {v}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
