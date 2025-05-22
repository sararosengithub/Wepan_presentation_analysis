import os
import sys
from pathlib import Path

if __name__ == "__main__":


    # Add the src directory to the Python path
    src_path = Path(__file__).resolve().parent
    sys.path.append(str(src_path))
relative_path = "/content/drive/MyDrive/Weapon_presentation/"


scenario_configurations = [
        {"id": "1", "event_file": relative_path + "2024.11.25.14.36.51.scenario1.event.evt", "frame_file": relative_path +"2024.11.25.14.36.51.scenario1.frame.csv"},
        {"id": "2", "event_file": relative_path + "2024.11.25.15.11.54.scenario2.event.evt", "frame_file": relative_path +"2024.11.25.15.11.54.scenario2.frame.csv"},
        {"id": "3", "event_file": relative_path + "2024.11.25.15.14.45.scenario3.event.evt", "frame_file": relative_path +"2024.11.25.15.14.45.scenario3.frame.csv"},
    ]

#WEAPON_POS_COLS = ['weapon_muzzle_position_x', 'weapon_muzzle_position_y', 'weapon_muzzle_position_z']
WEAPON_POS_COLS = ['controller_position_x', 'controller_position_y', 'controller_position_z']


# Define all thresholds in one place
ALL_THRESHOLDS = {
    'windowing': {
        'MOVEMENT_VELOCITY_THRESHOLD': 0.15, # m/s
        'SUSTAINED_MOVEMENT_FRAMES': 3,
        'MIN_Y_LIFT_FOR_START': 0.03, # meters
        'STABILIZATION_VELOCITY_THRESHOLD': 0.1, # m/s
        'SUSTAINED_STABILIZATION_FRAMES': 3,
    },
  'curve': {
        'LINE_RATIO_THRESHOLD': 1.15, 
        'MAX_WOBBLE_DEVIATION_M': 0.05, # NEW: e.g., 5cm max deviation from 3D line for "line" type
        'VERTICAL_DEVIATION_THRESHOLD_M': 0.05, # NEW: e.g., 5cm for Y deviation
        'LATERAL_DEVIATION_THRESHOLD_M': 0.05,  # NEW: e.g., 5cm for X deviation
        'R2_LINEAR_THRESHOLD': 0.98,  # R-squared for YZ linear fit to be considered "line"
        'R2_PARABOLIC_IMPROVEMENT_FACTOR': 1.05, # Parabolic R2 must be 5% > Linear R2
        'MIN_POINTS_FOR_FIT': 5,       # Minimum trajectory points for fitting
        'LINE_RATIO_FALLBACK': 1.15,  # motion_length/direct_dist for fallback line classification
    }
}
PLOT_OUTPUT_DIR_BASE = "all_scenario_plots_for_threshold_eval"
    