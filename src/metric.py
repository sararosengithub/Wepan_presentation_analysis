import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

def calculate_presentation_speed(T_pres_start, T_pres_end):
    """Calculates the speed (duration) of the presentation."""
    if T_pres_start is None or T_pres_end is None:
        return None
    return T_pres_end - T_pres_start

def calculate_presentation_motion(trajectory_df, weapon_pos_cols):
    """
    Calculates the total path length (motion) and direct distance of the presentation.
    Returns:
        tuple: (motion_length_m, direct_dist_m) in meters.
    """
    if trajectory_df.empty or len(trajectory_df) < 2:
        return 0.0, 0.0

    motion_length_m = 0.0
    positions = trajectory_df[weapon_pos_cols].values
    for i in range(len(positions) - 1):
        p1 = positions[i]
        p2 = positions[i+1]
        motion_length_m += euclidean(p1, p2)

    P_start_traj = positions[0]
    P_end_traj = positions[-1]
    direct_dist_m = euclidean(P_start_traj, P_end_traj)

    return motion_length_m, direct_dist_m


# --- PRESENTATION WINDOW AND TRAJECTORY FINDER ---
def find_presentation_window_and_trajectory(drill_frames_segment_with_velocity, weapon_pos_cols, T_drill_start_ts, window_thresholds):
    """
    Finds T_pres_start, T_pres_end and extracts the trajectory_df.
    `drill_frames_segment_with_velocity` should already have 'velocity_mag' and 'velocity_xyz' columns.
    Returns:
        tuple: (T_pres_start, T_pres_end, trajectory_df) or (None, None, pd.DataFrame()) if failed.
    """
    df = drill_frames_segment_with_velocity 

    T_pres_start = None
    idx_pres_start = -1
    initial_y_position_at_drill_start = df.loc[df['timestamp'] == T_drill_start_ts, weapon_pos_cols[1]].iloc[0] \
                                       if not df[df['timestamp'] == T_drill_start_ts].empty else df.iloc[0][weapon_pos_cols[1]]


    for i in range(len(df) - window_thresholds['SUSTAINED_MOVEMENT_FRAMES'] + 1):
        current_segment = df.iloc[i : i + window_thresholds['SUSTAINED_MOVEMENT_FRAMES']]
        if not (current_segment['velocity_mag'] >= window_thresholds['MOVEMENT_VELOCITY_THRESHOLD']).all():
            continue

        y_at_potential_start = df.iloc[i][weapon_pos_cols[1]]
        y_at_end_of_sustained_move = df.iloc[i + window_thresholds['SUSTAINED_MOVEMENT_FRAMES'] - 1][weapon_pos_cols[1]]

        # Access velocity_xyz from the DataFrame directly
        avg_vy = df['velocity_xyz'].iloc[i : i + window_thresholds['SUSTAINED_MOVEMENT_FRAMES']].apply(lambda v: v[1]).mean()


        if (y_at_potential_start >= initial_y_position_at_drill_start + window_thresholds['MIN_Y_LIFT_FOR_START'] or \
            y_at_end_of_sustained_move >= initial_y_position_at_drill_start + window_thresholds['MIN_Y_LIFT_FOR_START']) and \
           avg_vy > 0.01 :

            drops_back = False
            check_further_frames = min(len(df), i + window_thresholds['SUSTAINED_MOVEMENT_FRAMES'] + 3)
            for k_idx in range(i + window_thresholds['SUSTAINED_MOVEMENT_FRAMES'], check_further_frames):
                if df.iloc[k_idx][weapon_pos_cols[1]] < initial_y_position_at_drill_start + (window_thresholds['MIN_Y_LIFT_FOR_START'] / 2):
                    drops_back = True
                    break
            if not drops_back:
                idx_pres_start = df.index[i]
                T_pres_start = df.loc[idx_pres_start, 'timestamp']
                break

    if T_pres_start is None:
        print("  Analysis Detail: Could not determine presentation_start.")
        return None, None, pd.DataFrame()

    T_pres_end = None
    idx_pres_end = -1
    search_df_for_end = df[df['timestamp'] >= T_pres_start].copy()

    if len(search_df_for_end) < window_thresholds['SUSTAINED_STABILIZATION_FRAMES']:
         if not search_df_for_end.empty:
            idx_pres_end = search_df_for_end.index[-1]
            T_pres_end = search_df_for_end.loc[idx_pres_end, 'timestamp']
         else:
            return T_pres_start, None, pd.DataFrame()
    else:
        for i in range(len(search_df_for_end) - window_thresholds['SUSTAINED_STABILIZATION_FRAMES'] + 1):
            current_segment = search_df_for_end.iloc[i : i + window_thresholds['SUSTAINED_STABILIZATION_FRAMES']]
            if (current_segment['velocity_mag'] <= window_thresholds['STABILIZATION_VELOCITY_THRESHOLD']).all():
                idx_pres_end = current_segment.index[0]
                T_pres_end = search_df_for_end.loc[idx_pres_end, 'timestamp']
                break

    if T_pres_end is None:
        if not search_df_for_end.empty:
            idx_pres_end = search_df_for_end.index[-1]
            T_pres_end = search_df_for_end.loc[idx_pres_end, 'timestamp']
        else:
            return T_pres_start, None, pd.DataFrame()

    if T_pres_end <= T_pres_start:
        valid_end_frames = df[df['timestamp'] > T_pres_start]
        if not valid_end_frames.empty:
             idx_pres_end = valid_end_frames.index[0]
             T_pres_end = df.loc[idx_pres_end, 'timestamp']
        else:
             return T_pres_start, T_pres_start, pd.DataFrame() # No valid end after start

    trajectory_df = df[(df['timestamp'] >= T_pres_start) & (df['timestamp'] <= T_pres_end)]
    return T_pres_start, T_pres_end, trajectory_df

