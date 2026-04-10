"""
Open each wallpaper group PLY one at a time in Polyscope for screenshot capture.
Close the Polyscope window to advance to the next group.

Usage:
    python show_wallpaper_3d.py                  # upper garment (default)
    python show_wallpaper_3d.py --garment lower
    python show_wallpaper_3d.py --group stripes  # single group
"""
import argparse
import os
from visualize_simulation import visualize

GROUPS = ["stripes", "diagonal_stripes", "grid", "p4", "p4m", "pg", "pmg", "pgg"]
DATA_ROOT = "results/wallpaper_groups"
PERIOD_MM = 50.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--garment", default="upper", choices=["upper", "lower"])
    p.add_argument("--group", default=None, choices=GROUPS,
                   help="Show a single group instead of all 8")
    p.add_argument("--period", type=float, default=PERIOD_MM)
    return p.parse_args()


def main():
    args = parse_args()
    groups = [args.group] if args.group else GROUPS

    for group in groups:
        ply = os.path.join(DATA_ROOT, args.garment, group, "cloth_00000.ply")
        if not os.path.exists(ply):
            print(f"[skip] {group}: PLY not found at {ply}")
            continue
        print(f"\n[show] {group}  ({ply})")
        print("       Close the Polyscope window to continue to the next group.")
        visualize(ply_path=ply, period_u_mm=args.period,
                  period_v_mm=args.period, wallpaper_group=group)

    print("\nDone.")


if __name__ == "__main__":
    main()
