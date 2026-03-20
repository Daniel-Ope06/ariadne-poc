import os
# import numpy as np

from modules.target_detector import TargetDetector
from modules.color_segmenter import ColorSegmenter


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    IMG_PATH = os.path.join(PROJECT_ROOT, "data_generation",
                            "synthetic_dataset", "images", "maze_env_0000.png")
    LBL_PATH = os.path.join(PROJECT_ROOT, "data_generation",
                            "synthetic_dataset", "labels", "maze_env_0000.txt")

    GRID_RESOLUTION = 30
    detector = TargetDetector(grid_size=GRID_RESOLUTION)
    segmenter = ColorSegmenter(grid_size=GRID_RESOLUTION)

    print("\n--- Target Detection ---")
    try:
        targets = detector.fake_detect(IMG_PATH, LBL_PATH)
        for cls_id, data in targets.items():
            name = "UGV" if cls_id == 0 else "Human Target"
            print(f"[+] {name} Located:")
            print(f"    - Pixel Start (Top-Left): {data['bbox_top_left']}")
            print(f"    - Bounding Box Size: {data['bbox_size']}")
            print(f"    - Mapped to Matrix Node: {data['grid_node']}")
    except Exception as e:
        print(f"[!] Detection Error: {e}")
        return

    print("\n--- Color Segmentation ---")
    # segmenter.tune_thresholds(IMG_PATH)
    try:
        grid, original_img = segmenter.generate_matrix(IMG_PATH)
        print(f"[+] Matrix successfully generated! Shape: {grid.shape}")

        print("\n[+] Snippet of the Top-Left Matrix (10x10):")
        print(grid[0:10, 0:10])

    except Exception as e:
        print(f"[!] Segmentation Error: {e}")


if __name__ == "__main__":
    main()
