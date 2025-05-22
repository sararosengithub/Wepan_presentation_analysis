from .metric import calculate_presentation_speed, calculate_presentation_motion, find_presentation_window_and_trajectory 
from .classify_curve import  classify_presentation_curve, classify_presentation_curve_polynomial_fit
from .plot import plot_drill_presentation, plot_drill_presentation_user_focused
import numpy as np
import pandas as pd

def analyze_single_drill_presentation(drill_frames_segment, weapon_pos_cols, T_drill_start_ts, all_thresholds):
    """
    Analyzes a segment of frame data for a single drill's weapon presentation
    by calling specific metric calculation functions.
    """
    if drill_frames_segment.empty or len(drill_frames_segment) < 2:
        print("  Analysis: Not enough frame data for this drill segment.")
        return None, None, None, None,None, None, pd.DataFrame() # speed, motion, curve,curve_fit_polynomial, T_start, T_end, trajectory

    df = drill_frames_segment.copy()

    # Pre-calculate velocity once
    pos_diff = df[weapon_pos_cols].diff().fillna(0) # Fill NaN for first row
    time_diff_ms = df['timestamp'].diff()
    time_diff_s = time_diff_ms / 1000.0
    time_diff_s.iloc[0] = 1e-6 # Avoid division by zero for first frame, make it non-zero

    # Replace any subsequent zeros with a very small number to avoid division by zero
    time_diff_s[time_diff_s == 0] = 1e-6

    pos_diff_values = pos_diff.values # This is a 2D numpy array (N_frames, N_dims)
    time_diff_s_values = time_diff_s.values.reshape(-1, 1) # Reshape to (N_frames, 1) for broadcasting
    # Perform element-wise division.
    # Each row of pos_diff_values will be divided by the corresponding scalar in time_diff_s_values

    velocity_xyz_values = pos_diff_values / time_diff_s_values
    df['velocity_xyz'] = list(velocity_xyz_values)



    #df['velocity_xyz'] = list(pos_diff.apply(lambda row: row.values, axis=1) / time_diff_s.values[:, np.newaxis])

    df['velocity_mag'] = np.linalg.norm(df['velocity_xyz'].tolist(), axis=1)
    df.iloc[0, df.columns.get_loc('velocity_mag')] = 0 # Ensure first frame velocity is 0
    if not df.empty:
        df.at[df.index[0], 'velocity_xyz'] = np.array([0.0, 0.0, 0.0])

    T_pres_start, T_pres_end, trajectory_df = find_presentation_window_and_trajectory(
        df, weapon_pos_cols, T_drill_start_ts, all_thresholds['windowing']
    )

    if T_pres_start is None or T_pres_end is None or trajectory_df.empty or len(trajectory_df) < 2:
        print("  Analysis Detail: Failed to determine valid presentation window or trajectory.")
        return None, None, None, T_pres_start, T_pres_end, trajectory_df

    # Calculate metrics using the trajectory
    speed_ms = calculate_presentation_speed(T_pres_start, T_pres_end)

    motion_length_m, direct_dist_m = calculate_presentation_motion(trajectory_df, weapon_pos_cols)
    motion_length_cm = motion_length_m * 100.0

    curve_type = classify_presentation_curve(trajectory_df, weapon_pos_cols, motion_length_m, direct_dist_m, all_thresholds['curve'])

    curve_fit_polynomial, r2_lin_yz, r2_par_yz = classify_presentation_curve_polynomial_fit(trajectory_df, weapon_pos_cols, all_thresholds['curve'])
    print(f"curve_fit_polynomial={curve_fit_polynomial},  Fit R2s: Linear YZ={r2_lin_yz:.3f}, Parabolic YZ={r2_par_yz:.3f}")

    print(f"  Analysis Results: Speed={speed_ms if speed_ms is not None else 'N/A'}ms, Motion={motion_length_cm:.2f}cm, Curve={curve_type}, PresStart={T_pres_start}, PresEnd={T_pres_end}")
    return speed_ms, motion_length_cm, curve_type, curve_fit_polynomial, T_pres_start, T_pres_end, trajectory_df


# --- NEW: Function to analyze drills for a loaded scenario ---
def analyze_scenario_drills(scenario_id, frame_df, drills_to_analyze, weapon_pos_cols, thresholds, plot_output_dir_base):
    print(f"\n--- Analyzing Drills for Scenario: {scenario_id} ---")
    scenario_drill_metrics = []

    if frame_df.empty or not drills_to_analyze:
        print(f"  No frame data or drills to analyze for scenario {scenario_id}.")
        return scenario_drill_metrics

    for drill_data in drills_to_analyze:
        drill_uid = drill_data['drill_uid']
        T_drill_start = drill_data['start_time']
        T_first_shot = drill_data.get('first_shot_time')
        print(f"\n  Analyzing Drill: {drill_uid} (Start: {T_drill_start}, First Shot: {T_first_shot}) for Scenario {scenario_id}")
        
        error_msg = None
        if T_first_shot is None: error_msg = "Missing T_first_shot"
        elif T_first_shot <= T_drill_start: error_msg = "T_first_shot not after T_drill_start"
        if error_msg:
            print(f"    Skipping: {error_msg}")
            scenario_drill_metrics.append({"scenario_id": scenario_id, "drill_uid": drill_uid, "error": error_msg})
            continue

        drill_frames_segment = frame_df[
            (frame_df['timestamp'] >= T_drill_start) & (frame_df['timestamp'] < T_first_shot)
        ].copy()
        if drill_frames_segment.empty or len(drill_frames_segment) < 2:
            print(f"    Skipping: Not enough frame data for this drill segment.")
            scenario_drill_metrics.append({"scenario_id": scenario_id, "drill_uid": drill_uid, "error": "Insufficient frame data"})
            continue

        analysis_output = analyze_single_drill_presentation(
            drill_frames_segment, weapon_pos_cols, T_drill_start, thresholds
        )
        speed, motion, curve, curve_fit_polynomial, T_start_calc, T_end_calc, trajectory_df_for_plot = analysis_output
        
        result_entry = {"scenario_id": scenario_id, "drill_uid": drill_uid,
                        "T_drill_start_event": T_drill_start, "T_first_shot_event": T_first_shot,
                        "T_presentation_start_calc": T_start_calc, "T_presentation_end_calc": T_end_calc, "error": None}
        if speed is not None:
            result_entry.update({"speed_ms": speed, "motion_length_cm": motion, "curve_type": curve, "curve_fit_polynomial": curve_fit_polynomial})
            if not trajectory_df_for_plot.empty:
                plot_drill_presentation(scenario_id, drill_uid, trajectory_df_for_plot, T_start_calc, T_end_calc, weapon_pos_cols, curve, curve_fit_polynomial, plot_output_dir_base)
                plot_drill_presentation_user_focused(
                    scenario_id, 
                    drill_uid, 
                    trajectory_df_for_plot, 
                    T_start_calc, # Timestamp of presentation start
                    T_end_calc,   # Timestamp of presentation end
                    weapon_pos_cols, 
                    curve,        # The classified curve type
                    speed,        # Calculated speed
                    motion,       # Calculated motion length
                    plot_output_dir_base # Your base directory for plots
                )
        else:
            result_entry["error"] = "Presentation analysis failed"
            if T_start_calc is not None and not trajectory_df_for_plot.empty:
                 plot_drill_presentation(scenario_id, drill_uid, trajectory_df_for_plot, T_start_calc, T_end_calc, weapon_pos_cols, "error_in_analysis", "error_in_analysis", plot_output_dir_base)
                 plot_drill_presentation_user_focused(
                    scenario_id, drill_uid, trajectory_df_for_plot, 
                    T_start_calc, T_end_calc, # T_end_calc might be None if only start was found
                    weapon_pos_cols, "error_in_analysis", 
                    None, None, # Speed and motion might be None
                    plot_output_dir_base
                 )
        scenario_drill_metrics.append(result_entry)
        
    return scenario_drill_metrics
