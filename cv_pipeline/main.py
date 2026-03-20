import os
import cv2
# import numpy as np

from modules.target_detector import TargetDetector
from modules.color_segmenter import ColorSegmenter
from modules.path_finder import PathFinder
from modules.visualizer import Visualizer


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    IMG_PATH = os.path.join(PROJECT_ROOT, "data_generation",
                            "synthetic_dataset", "images", "maze_env_0000.png")
    LBL_PATH = os.path.join(PROJECT_ROOT, "data_generation",
                            "synthetic_dataset", "labels", "maze_env_0000.txt")

    GRID_RESOLUTION = 30
    detector = TargetDetector(grid_size=GRID_RESOLUTION)
    segmenter = ColorSegmenter(grid_size=GRID_RESOLUTION)
    path_finder = PathFinder()
    visualizer = Visualizer(grid_size=GRID_RESOLUTION)

    print("\n--- Target Detection ---")
    try:
        targets = detector.fake_detect(IMG_PATH, LBL_PATH)

        ugv_data = targets.get(0)
        human_data = targets.get(1)

        if not ugv_data or not human_data:
            print("Error: Could not locate both UGV and Human ")
            return

        start_node = ugv_data['grid_node']
        goal_node = human_data['grid_node']

        print(f"[+] UGV located at matrix node: {start_node}")
        print(f"[+] Human located at matrix node: {goal_node}")
    except Exception as e:
        print(f"[!] Detection Error: {e}")
        return

    print("\n--- Color Segmentation ---")
    # segmenter.tune_thresholds(IMG_PATH)
    try:
        binary_matrix = segmenter.generate_matrix(IMG_PATH)
        print(f"Matrix Shape: {binary_matrix.shape}")
        # print("\n\n", binary_matrix, "\n\n")
    except Exception as e:
        print(f"[!] Segmentation Error: {e}")
        return

    print("\n--- Path Finding ---")
    try:
        print("[*] Calculating optimal route...")
        optimal_path = path_finder.find_path(
            binary_matrix, start_node, goal_node)

        if not optimal_path:
            print("[!] ROUTING FAILED: No valid path exists.")
        else:
            print(f"Path found in {len(optimal_path)} steps!")
            print(f"(First 5 steps): {optimal_path[:5]} ...")
            print(f"(Last 5 steps): ... {optimal_path[-5:]}")
    except Exception as e:
        print(f"[!] Pathfinding Error: {e}")

    print("\n--- Visualization ---")
    try:
        final_output = visualizer.draw_results(
            IMG_PATH, targets, optimal_path)  # type: ignore

        filename = os.path.basename(IMG_PATH)
        save_path = os.path.join(OUTPUT_DIR, f"solved_{filename}")
        cv2.imwrite(save_path, final_output)  # type: ignore
        print(f"[+] Image successfully saved to: {save_path}")

        print("[+] Rendering complete. Opening display window...")
        cv2.imshow("Routing Output", final_output)  # type: ignore

        # Keep the window open until the user presses any key
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[!] Visualization Error: {e}")


if __name__ == "__main__":
    main()
