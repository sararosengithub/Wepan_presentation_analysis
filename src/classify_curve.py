import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from .classify_curve import calculate_presentation_motion

def classify_presentation_curve(trajectory_df, weapon_pos_cols, motion_length_m, direct_dist_m, curve_thresholds):
    """
    Classifies the curve shape of the presentation based on revised definitions.
    Args:
        trajectory_df (pd.DataFrame): DataFrame of the presentation trajectory.
        weapon_pos_cols (list): Names of X, Y, Z columns.
        motion_length_m (float): Total path length of the trajectory in meters.
        direct_dist_m (float): Straight-line distance between start and end of trajectory in meters.
        curve_thresholds (dict): Dictionary containing thresholds like:
            'LINE_RATIO_THRESHOLD': Max ratio of motion_length / direct_dist for a "line".
            'VERTICAL_DEVIATION_THRESHOLD_M': Max deviation in Y (meters) from direct path to still be "line".
                                            Also used to quantify "above/below" for push/swing.
            'LATERAL_DEVIATION_THRESHOLD_M': Max deviation in X (meters) from direct path (in XZ) to be "line".
                                           Also used to quantify "side-to-side" for push/swing.
    Returns:
        str: "line", "push", "swing", or "other".
    """
    VERTICAL_DEVIATION_THRESHOLD_M = curve_thresholds['VERTICAL_DEVIATION_THRESHOLD_M']
    LATERAL_DEVIATION_THRESHOLD_M = curve_thresholds['LATERAL_DEVIATION_THRESHOLD_M']

    if trajectory_df.empty or len(trajectory_df) < 2:
        return "other"

    positions = trajectory_df[weapon_pos_cols].values
    P_start = positions[0]
    P_end = positions[-1]

    # 1. Check for "Line"
    is_line_candidate = False
    if direct_dist_m < 0.01: 
        is_line_candidate = True
    elif (motion_length_m / direct_dist_m) < curve_thresholds['LINE_RATIO_THRESHOLD']:
        is_line_candidate = True

    if is_line_candidate:
        max_total_deviation_m = 0.0
        if direct_dist_m > 1e-6: # Avoid division by zero if P_start and P_end are same
            v = P_end - P_start
            v_mag_sq = np.dot(v, v)
            for P_point in positions[1:-1]: 
                w = P_point - P_start

                t = np.dot(w, v) / v_mag_sq
                if 0 <= t <= 1: # Projection falls within the segment
                    P_proj = P_start + t * v
                    deviation = euclidean(P_point, P_proj)
                    if deviation > max_total_deviation_m:
                        max_total_deviation_m = deviation
                else: # Projection is outside segment, check distance to P_start or P_end
                    dev_to_start = euclidean(P_point, P_start)
                    dev_to_end = euclidean(P_point, P_end)
                    deviation = min(dev_to_start, dev_to_end) # A bit of a simplification here
                    if deviation > max_total_deviation_m:
                        max_total_deviation_m = deviation


        # If max_total_deviation is small enough, it's a line.

        if max_total_deviation_m < curve_thresholds.get('MAX_WOBBLE_DEVIATION_M', 0.05): # e.g., 5cm wobble
             return "line"
        # else it was a candidate but too wobbly, falls through to push/swing/other


    # 2. If not a "Line", analyze for "Pushing" or "Swinging" based on deviations

    # Vertical deviation (Y-axis)
    # For each point in trajectory, calculate its Y vs the Y of the ideal line at that XZ projection
    max_y_above_line = 0.0 # Max positive (Y_actual - Y_on_line)
    max_y_below_line = 0.0 # Max positive (Y_on_line - Y_actual) where Y_actual < Y_on_line

    # Lateral deviation (X-axis, relative to XZ line)
    max_x_deviation_abs = 0.0 # Max absolute X deviation from XZ line

    if direct_dist_m > 1e-6: # Only if there's a defined line
        # Vector from P_start to P_end
        direction_vector = P_end - P_start

        for P_current in positions[1:-1]: # Intermediate points
            # Parameter t for projection onto the line P_start -> P_end
            # t = dot(P_current - P_start, direction_vector) / dot(direction_vector, direction_vector)
            # P_on_line = P_start + t * direction_vector


            # Or, project P_current onto the line to find the closest point P_on_line.

            # Vector from P_start to P_current
            w = P_current - P_start
            # Projection parameter (t)
            t = np.dot(w, direction_vector) / np.dot(direction_vector, direction_vector)

            # Clamp t to [0, 1] to ensure P_on_line is on the segment for deviation checks
            t_clamped = max(0, min(1, t))
            P_on_line_segment = P_start + t_clamped * direction_vector

            # Vertical deviation
            y_actual = P_current[1]
            y_on_line = P_on_line_segment[1]
            vertical_diff = y_actual - y_on_line
            if vertical_diff > 0: # Actual path is above the direct line's Y
                max_y_above_line = max(max_y_above_line, vertical_diff)
            else: # Actual path is at or below the direct line's Y
                max_y_below_line = max(max_y_below_line, abs(vertical_diff))


            # Lateral (X) deviation in XZ plane
            # Line in XZ plane: P_start_xz -> P_end_xz
            P_start_xz = P_start[[0, 2]]
            P_end_xz = P_end[[0, 2]]
            P_current_xz = P_current[[0, 2]]

            if not np.allclose(P_start_xz, P_end_xz):
                A = P_end_xz[1] - P_start_xz[1]   # dz
                B = P_start_xz[0] - P_end_xz[0]   # -dx
                C = P_end_xz[0] * P_start_xz[1] - P_end_xz[1] * P_start_xz[0] # x2*z1 - z2*x1
                if not (A == 0 and B == 0):
                    dist_to_xz_line = abs(A * P_current_xz[0] + B * P_current_xz[1] + C) / np.sqrt(A**2 + B**2)
                    max_x_deviation_abs = max(max_x_deviation_abs, dist_to_xz_line)

    # Decision logic based on deviations:
    VERTICAL_DEV_THRESH = VERTICAL_DEVIATION_THRESHOLD_M
    LATERAL_DEV_THRESH = LATERAL_DEVIATION_THRESHOLD_M

    is_push_y = max_y_above_line > VERTICAL_DEV_THRESH
    is_swing_y = max_y_below_line > VERTICAL_DEV_THRESH
    is_side_to_side_x = max_x_deviation_abs > LATERAL_DEV_THRESH


    if is_push_y and not is_swing_y: 
        return "push"
    if is_swing_y and not is_push_y: 
        return "swing"

    # If both Y deviations are present or neither are dominant over the threshold,
    # or if only X deviation is significant:
    if is_side_to_side_x:
        if max_y_above_line > max_y_below_line and max_y_above_line > VERTICAL_DEV_THRESH * 0.5: # Slight tendency to be above
            return "push" 
        elif max_y_below_line > max_y_above_line and max_y_below_line > VERTICAL_DEV_THRESH * 0.5: # Slight tendency to be below
            return "swing" 
        else: # No clear Y tendency with the side-to-side, or Y deviations are small.
            return "other" 
    # If it wasn't a line, and didn't meet push/swing criteria significantly:
    return "other"



def fit_and_evaluate_polynomial(x_coords, y_coords, degree):
    """Fits a polynomial and returns coefficients and R-squared."""
    if len(x_coords) < degree + 1: # Not enough points to fit
        return None, -1 

    try:
        coeffs = np.polyfit(x_coords, y_coords, degree)
        poly_model = np.poly1d(coeffs)
        y_pred = poly_model(x_coords)
        r_squared = r2_score(y_coords, y_pred)
        return coeffs, r_squared
    except (np.linalg.LinAlgError, ValueError): 
        return None, -1



def classify_presentation_curve_polynomial_fit(trajectory_df, weapon_pos_cols, curve_thresholds):
    """
    Classifies the curve shape using polynomial fitting and R-squared.
    curve_thresholds should include:
        'R2_LINEAR_THRESHOLD': R-squared value above which a linear fit is considered "line". (e.g., 0.98)
        'R2_PARABOLIC_IMPROVEMENT_FACTOR': How much better parabolic R2 needs to be than linear R2. (e.g., 1.05 for 5% better)
        'MIN_POINTS_FOR_FIT': Minimum number of points for reliable fitting (e.g., 5)
        'LATERAL_DEVIATION_THRESHOLD_M': For "side-to-side" aspect (can still be used).
    """
    if trajectory_df.empty or len(trajectory_df) < curve_thresholds.get('MIN_POINTS_FOR_FIT', 2):
        print(f"trajectory_df to classify curve has :{len(trajectory_df)} points")
        return "other", -1, -1 # curve_type, r2_linear_yz, r2_parabolic_yz

    positions = trajectory_df[weapon_pos_cols].values.astype(float)
    P_start = positions[0]
    P_end = positions[-1]

    # Primary analysis in YZ plane (Side View: Y as function of Z)
    # Y is weapon_pos_cols[1], Z is weapon_pos_cols[2]
    z_coords_yz = positions[:, 2] # Independent variable for side view
    y_coords_yz = positions[:, 1] # Dependent variable

    # Ensure there's variation in the independent variable for a meaningful fit
    if np.std(z_coords_yz) < 1e-3 and np.std(y_coords_yz) < 1e-3: # Almost no movement in YZ
        return "line", 1.0, 1.0 # Effectively a point or tiny line
    if np.std(z_coords_yz) < 1e-3: # Vertical line (no change in Z), treat as line for y=f(z)
         # Could alternatively fit z = f(y) if Y variation is larger
        coeffs_linear_yz, r2_linear_yz = fit_and_evaluate_polynomial(y_coords_yz, z_coords_yz, 1)
        coeffs_parab_yz, r2_parab_yz = fit_and_evaluate_polynomial(y_coords_yz, z_coords_yz, 2)
        # Swap interpretation if we fitted Z = f(Y)
    else:
        coeffs_linear_yz, r2_linear_yz = fit_and_evaluate_polynomial(z_coords_yz, y_coords_yz, 1)
        coeffs_parab_yz, r2_parab_yz = fit_and_evaluate_polynomial(z_coords_yz, y_coords_yz, 2)


    # Top-Down analysis in XZ plane (Z as function of X, or X as function of Z)
    # X is weapon_pos_cols[0]
    x_coords_xz = positions[:, 0]
    z_coords_xz = positions[:, 2]
    
    r2_linear_xz = -1
    r2_parab_xz = -1

    # Decide whether to fit z=f(x) or x=f(z) based on which has more spread
    if np.std(x_coords_xz) > np.std(z_coords_xz) and np.std(x_coords_xz) > 1e-3 :
        # Fit z = f(x)
        _, r2_linear_xz = fit_and_evaluate_polynomial(x_coords_xz, z_coords_xz, 1)
        coeffs_parab_xz, r2_parab_xz = fit_and_evaluate_polynomial(x_coords_xz, z_coords_xz, 2)
        a_parab_xz = coeffs_parab_xz[0] if coeffs_parab_xz is not None and len(coeffs_parab_xz)==3 else 0
    elif np.std(z_coords_xz) > 1e-3:
        # Fit x = f(z)
        _, r2_linear_xz = fit_and_evaluate_polynomial(z_coords_xz, x_coords_xz, 1)
        coeffs_parab_xz, r2_parab_xz = fit_and_evaluate_polynomial(z_coords_xz, x_coords_xz, 2)
        a_parab_xz = coeffs_parab_xz[0] if coeffs_parab_xz is not None and len(coeffs_parab_xz)==3 else 0
    else: # Not much movement in XZ plane
        a_parab_xz = 0 # No lateral curve

    # --- Decision Logic ---
    # 1. Primary check: YZ plane for line vs vertical arc
    if r2_linear_yz >= curve_thresholds['R2_LINEAR_THRESHOLD']:
        # Strong linear fit in side view. Check if also linear in top-down.
        if r2_linear_xz >= curve_thresholds['R2_LINEAR_THRESHOLD'] or r2_linear_xz < 0: # or XZ is not very curved
             # Also check motion_length vs direct_dist for overall straightness as a fallback
            motion_len, direct_dist = calculate_presentation_motion(trajectory_df, weapon_pos_cols) # Recalculate here or pass in
            if direct_dist < 0.01 or (motion_len / direct_dist if direct_dist > 1e-9 else 1.0) < curve_thresholds.get('LINE_RATIO_FALLBACK', 1.15):
                return "line", r2_linear_yz, r2_parab_yz
    
    # 2. If YZ is not perfectly linear, see if parabolic is a much better fit
    #    A significantly better parabolic fit suggests an arc.
    yz_parabolic_better = (r2_parab_yz > r2_linear_yz * curve_thresholds['R2_PARABOLIC_IMPROVEMENT_FACTOR'] and \
                           r2_parab_yz > 0.85) # Ensure parabolic fit itself is decent

    if yz_parabolic_better:
        a_parab_yz = coeffs_parab_yz[0] if coeffs_parab_yz is not None and len(coeffs_parab_yz)==3 else 0
        
        # Determine overall Y direction (should always be up for presentation)
        y_start = y_coords_yz[0]
        y_end = y_coords_yz[-1]
        is_overall_up = y_end > y_start

        # Heuristic for Push/Swing based on parabola's concavity in YZ
        # And considering the overall upward movement of presentation.
        # If a_parab_yz < 0, parabola opens down. For an upward motion, this implies arcing *over* the chord.
        # If a_parab_yz > 0, parabola opens up. For an upward motion, this implies arcing *under* the chord.
        
        # Also consider lateral arcing from XZ plane
        xz_parabolic_significant = (r2_parab_xz > r2_linear_xz * curve_thresholds['R2_PARABOLIC_IMPROVEMENT_FACTOR'] and \
                                    r2_parab_xz > 0.80) # XZ parabolic is also a good fit

        if is_overall_up: # This should always be true for a weapon presentation
            if a_parab_yz < -1e-3: # Opens downwards (arcs above the chord) - "Push" tendency
                curve_type = "push"
            elif a_parab_yz > 1e-3: # Opens upwards (arcs below the chord) - "Swing" tendency
                curve_type = "swing"
            else: # Very little vertical curvature, might be mostly lateral
                curve_type = "other_yz_flat" # Tentative
        else: # Should not happen for presentation
            curve_type = "other_yz_down"

        # If there's also significant lateral arcing, the "side-to-side" aspect is present
        if xz_parabolic_significant:
            # The task says "push" or "swing" can have "side-to-side".
            # If YZ was flat, XZ curve determines push/swing based on which side it bulges.
            # This gets complex. For now, if YZ gave a push/swing, and XZ is also curved, it's a more pronounced error.
            if curve_type == "other_yz_flat": # If YZ was mostly straight but arced
                 # If a_parab_xz is significant, it's a lateral swing.
                 # We don't have a clear "push" vs "swing" from XZ concavity alone without target direction.
                 # So, if YZ is flat but XZ is arced, let's call it a general "swing" or "lateral_arc".
                 return "swing", r2_linear_yz, r2_parab_yz # Or "lateral_arc"
            else: # YZ already classified as push/swing, XZ adds to it
                return curve_type, r2_linear_yz, r2_parab_yz # Keep YZ classification

        # If YZ was arced but XZ was not significantly arced
        if curve_type not in ["other_yz_flat", "other_yz_down"]:
            return curve_type, r2_linear_yz, r2_parab_yz

    # Fallback if none of the above clear conditions are met
    # Could still use the deviation based logic here as a fallback or for "other"
    motion_len, direct_dist = calculate_presentation_motion(trajectory_df, weapon_pos_cols)
    if direct_dist < 0.01 or (motion_len / direct_dist if direct_dist > 1e-9 else 1.0) < curve_thresholds.get('LINE_RATIO_FALLBACK', 1.15):
        return "line", r2_linear_yz, r2_parab_yz # Fallback to line if path is short and efficient
        
    return "other", r2_linear_yz, r2_parab_yz