import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv

# --- EDA function for threshold exploration  ---
def explore_data_for_thresholds(scenario_id, frame_df, drills_to_analyze, weapon_pos_cols, output_dir_base):
    print(f"\n--- Exploratory Data Analysis for Thresholds (Scenario: {scenario_id}) ---")
    if frame_df.empty or not drills_to_analyze:
        print("  No data for threshold EDA.")
        return

    # Create scenario-specific output directory for EDA plots
    output_dir = os.path.join(output_dir_base, f"scenario_{scenario_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) # Use exist_ok=True to prevent error if dir already exists
    print(f" output_dir : {output_dir}")

    #  Plot velocity profile for the  drills
    # --------------------------------------------------
    y_col = weapon_pos_cols[1]
    all_velocities = []
    num_drills_to_plot = len(drills_to_analyze)
    for i in range(num_drills_to_plot):
        drill_data = drills_to_analyze[i]
        drill_uid = drill_data['drill_uid']
        T_drill_start = drill_data['start_time']
        T_first_shot = drill_data.get('first_shot_time')

        if T_first_shot is None or T_first_shot <= T_drill_start:
            continue

        segment = frame_df[(frame_df['timestamp'] >= T_drill_start) & (frame_df['timestamp'] < T_first_shot)].copy()
        if len(segment) < 2: continue

        # Simplified velocity calculation for plotting
        for col in weapon_pos_cols: segment[col] = pd.to_numeric(segment[col], errors='coerce')
        segment.dropna(subset=weapon_pos_cols, inplace=True)
        if len(segment) < 2: continue

        pos_diff = segment[weapon_pos_cols].diff().fillna(0)
        time_diff_s = (segment['timestamp'].diff() / 1000.0).fillna(1e-6)
        time_diff_s.iloc[0] = max(time_diff_s.iloc[0], 1e-6)
        time_diff_s[time_diff_s == 0] = 1e-6
        
        velocity_xyz_vals = pos_diff.values / time_diff_s.values.reshape(-1,1)
        segment['velocity_mag_eda'] = np.linalg.norm(velocity_xyz_vals, axis=1)
        if not segment.empty: segment.iloc[0, segment.columns.get_loc('velocity_mag_eda')] = 0.0
        all_velocities.extend(segment['velocity_mag_eda'].tolist())

        # Plot Velocity for this drill
        plt.figure(figsize=(12, 6))
        plt.plot(segment['timestamp'], segment['velocity_mag_eda'], label=f'Drill {drill_uid} Velocity')
        plt.title(f'Velocity Profile - Scenario {scenario_id}, Drill {drill_uid}')
        plt.xlabel('Timestamp')
        plt.ylabel('Velocity Magnitude (m/s)')
        plt.legend()
        plt.grid(True)

        safe_drill_uid = "".join(c if c.isalnum() or c in ('-', '_', '.') else "_" for c in drill_uid)
        plot_filename = os.path.join(output_dir, f"eda_threshold_Velocity_explore_drill_{safe_drill_uid}.png")

        try:
            plt.savefig(plot_filename)
            print(f"  Generated velocity plot for threshold exploration: {plot_filename}")
        except Exception as e:
             print(f"  Error saving EDA plot {plot_filename}: {e}")

        plt.show()
        print(f"  Generated velocity plot for threshold exploration: eda_threshold_explore_sc{scenario_id}_drill_{drill_uid}.png")

        # Plot Y-axis movement for this drill
        plt.figure(figsize=(12, 6))
        plt.plot(segment['timestamp'], segment[y_col], label=f'Drill {drill_uid} Y-Position')
        plt.title(f'Y-Axis Movement - Scenario {scenario_id}, Drill {drill_uid}')
        plt.xlabel('Timestamp'); plt.ylabel(f'{y_col} (meters)'); plt.legend(); plt.grid(True)

        safe_drill_uid = "".join(c if c.isalnum() or c in ('-', '_', '.') else "_" for c in drill_uid)
        plot_filename = os.path.join(output_dir, f"eda_threshold_Y_movement_sc{safe_drill_uid}.png")

        try:
            plt.savefig(plot_filename)
            print(f"  Generated velocity plot for threshold exploration: {plot_filename}")
        except Exception as e:
             print(f"  Error saving EDA plot {plot_filename}: {e}")

        plt.show()
        print(f"  Generated Y-axis movement plot for threshold exploration: eda_threshold_Y_movement_sc{scenario_id}_drill_{drill_uid}.png")
        

    # Plot histogram of all velocities
    plt.figure(figsize=(10,6))
    sns.histplot(all_velocities, bins=100, kde=True)
    plt.title("Histogram of all observed Velocity Magnitudes")
    plt.xlabel("Velocity (m/s)")
    plt.xlim(0, max(np.percentile(all_velocities, 99), 0.5)) # Zoom in on relevant range
    
    plot_filename = os.path.join(output_dir, f"histogram_velocities.png")
    try:
            plt.savefig(plot_filename)
            print(f"  Generated histogram of all velocities: {plot_filename}")
    except Exception as e:
             print(f"  Error saving EDA plot {plot_filename}: {e}")

    plt.show()

    
    print("--- End of Threshold EDA for this scenario ---")
    print("Review generated plots and adjust ALL_THRESHOLDS if needed before full analysis.")

