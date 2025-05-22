import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import os


def plot_drill_presentation(scenario_id, drill_uid, trajectory_df, T_pres_start, T_pres_end, weapon_pos_cols, curve_type,curve_fit_polynomial, base_output_dir="plots"):
    """
    Creates and saves a 2D plot (XZ and YZ views) of the weapon presentation path.
    """
    if trajectory_df.empty or len(trajectory_df) < 2:
        print(f"  Plotting: No trajectory data to plot for drill {drill_uid}.")
        return

    # Create scenario-specific output directory
    output_dir = os.path.join(base_output_dir, f"scenario_{scenario_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for col in weapon_pos_cols:
        trajectory_df[col] = pd.to_numeric(trajectory_df[col], errors='coerce')
    trajectory_df = trajectory_df.dropna(subset=weapon_pos_cols) # Drop rows if conversion failed

    if trajectory_df.empty or len(trajectory_df) < 2:
        print(f"  Plotting: Not enough valid numeric trajectory data after conversion for drill {drill_uid}.")
        return

    x_col, y_col, z_col = weapon_pos_cols[0], weapon_pos_cols[1], weapon_pos_cols[2]

    path_x = trajectory_df[x_col]
    path_y = trajectory_df[y_col]
    path_z = trajectory_df[z_col]

    # Find the actual start and end points on the plotted trajectory
    # (T_pres_start and T_pres_end are timestamps, need to map to trajectory points)
    start_point_data = trajectory_df[trajectory_df['timestamp'] == T_pres_start]
    end_point_data = trajectory_df[trajectory_df['timestamp'] == T_pres_end]

    start_x, start_y, start_z = None, None, None
    if not start_point_data.empty:
        start_x = start_point_data[x_col].iloc[0]
        start_y = start_point_data[y_col].iloc[0]
        start_z = start_point_data[z_col].iloc[0]
    elif not trajectory_df.empty: # Fallback to first point of trajectory if exact timestamp match fails
        start_x, start_y, start_z = path_x.iloc[0], path_y.iloc[0], path_z.iloc[0]


    end_x, end_y, end_z = None, None, None
    if not end_point_data.empty:
        end_x = end_point_data[x_col].iloc[0]
        end_y = end_point_data[y_col].iloc[0]
        end_z = end_point_data[z_col].iloc[0]
    elif not trajectory_df.empty: # Fallback to last point
        end_x, end_y, end_z = path_x.iloc[-1], path_y.iloc[-1], path_z.iloc[-1]


    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Drill: {drill_uid} - Presentation Path (Type: {curve_type}, Type fit to parabolic: {curve_fit_polynomial})", fontsize=16)

    # Top-Down View (XZ plane)
    axs[0].plot(path_x, path_z, marker='.', linestyle='-', label="Path")
    if start_x is not None:
        axs[0].plot(start_x, start_z, 'go', markersize=10, label="Start (T_pres_start)")
    if end_x is not None:
        axs[0].plot(end_x, end_z, 'ro', markersize=10, label="End (T_pres_end)")
    axs[0].set_xlabel("X-coordinate (meters)")
    axs[0].set_ylabel("Z-coordinate (meters, Forward)")
    axs[0].set_title("Top-Down View (XZ Plane)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal') # Equal scaling for X and Z

    # Side View (YZ plane)
    axs[1].plot(path_z, path_y, marker='.', linestyle='-', label="Path") # Z on X-axis for intuitive "forward"
    if start_z is not None:
        axs[1].plot(start_z, start_y, 'go', markersize=10, label="Start (T_pres_start)")
    if end_z is not None:
        axs[1].plot(end_z, end_y, 'ro', markersize=10, label="End (T_pres_end)")
    axs[1].set_xlabel("Z-coordinate (meters, Forward)")
    axs[1].set_ylabel("Y-coordinate (meters, Up)")
    axs[1].set_title("Side View (ZY Plane)")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].axis('equal') # Equal scaling for Z and Y

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Sanitize drill_uid for filename
    safe_drill_uid = "".join(c if c.isalnum() else "_" for c in drill_uid)
    plot_filename = os.path.join(output_dir, f"drill_{safe_drill_uid}_presentation.png")
    try:
        plt.savefig(plot_filename)
        print(f"  Plot saved: {plot_filename}")
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")
    plt.close(fig) 




def plot_drill_presentation_user_focused(
    scenario_id, drill_uid, 
    trajectory_df, # DataFrame with X,Y,Z columns for the presentation path
    T_pres_start, T_pres_end, # Timestamps of calculated start and end
    weapon_pos_cols, 
    curve_type, 
    speed_ms, # Pass in the calculated speed
    motion_length_cm, # Pass in the calculated motion length
    base_output_dir="plots_user"
):
    """
    Creates and saves a user-focused 2D plot of the weapon presentation path.
    """
    if trajectory_df.empty or len(trajectory_df) < 2:
        print(f"  Plotting User (Scen {scenario_id}, Drill {drill_uid}): No trajectory data.")
        return

    output_dir = os.path.join(base_output_dir, f"scenario_{scenario_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for col in weapon_pos_cols:
        trajectory_df[col] = pd.to_numeric(trajectory_df[col], errors='coerce')
    trajectory_df = trajectory_df.dropna(subset=weapon_pos_cols)
    if trajectory_df.empty or len(trajectory_df) < 2:
        print(f"  Plotting User (Scen {scenario_id}, Drill {drill_uid}): Not enough valid numeric data."); return

    x_col, y_col, z_col = weapon_pos_cols[0], weapon_pos_cols[1], weapon_pos_cols[2]
    path_x = trajectory_df[x_col].values
    path_y = trajectory_df[y_col].values
    path_z = trajectory_df[z_col].values

 
    actual_start_coords = trajectory_df.iloc[0][weapon_pos_cols].astype(float).values
    actual_end_coords = trajectory_df.iloc[-1][weapon_pos_cols].astype(float).values
    
    start_x, start_y, start_z = actual_start_coords[0], actual_start_coords[1], actual_start_coords[2]
    end_x, end_y, end_z = actual_end_coords[0], actual_end_coords[1], actual_end_coords[2]


    fig, axs = plt.subplots(1, 2, figsize=(16, 7)) # Slightly larger
    title_text = (f"Drill: {drill_uid} - Presentation Analysis\n"
                  f"Type: {curve_type}, Speed: {speed_ms:.0f}ms, Motion Path: {motion_length_cm:.1f}cm")
    fig.suptitle(title_text, fontsize=14)

    # --- Top-Down View (XZ plane) ---
    ax_xz = axs[0]
    ax_xz.plot(path_x, path_z, marker='.', linestyle='-', label="Your Path", zorder=2)
    ax_xz.plot([start_x, end_x], [start_z, end_z], 'k--', label="Direct Path", zorder=1) # Ideal direct path
    ax_xz.plot(start_x, start_z, 'go', markersize=10, label="Start", zorder=3)
    ax_xz.plot(end_x, end_z, 'ro', markersize=10, label="End", zorder=3)

    # Find and show max lateral deviation
    max_lateral_dev_val = 0
    max_lateral_dev_idx = -1
    if not np.allclose(actual_start_coords[[0,2]], actual_end_coords[[0,2]]): # If start and end XZ are not the same
        A = end_z - start_z       # dz
        B = start_x - end_x       # -dx
        C = end_x * start_z - end_z * start_x # x2*z1 - z2*x1
        if not (A == 0 and B == 0):
            for idx in range(len(path_x)):
                dist = abs(A * path_x[idx] + B * path_z[idx] + C) / np.sqrt(A**2 + B**2)
                if dist > max_lateral_dev_val:
                    max_lateral_dev_val = dist
                    max_lateral_dev_idx = idx
    if max_lateral_dev_idx != -1 and max_lateral_dev_val > 0.01: # Only show if deviation > 1cm
        ax_xz.plot(path_x[max_lateral_dev_idx], path_z[max_lateral_dev_idx], 'bo', markersize=7, label=f"Max Lateral Dev ({max_lateral_dev_val*100:.1f}cm)", zorder=3)
        # Arrow from ideal line to point of max deviation
        # Project point onto line
        t = ((path_x[max_lateral_dev_idx] - start_x) * (end_x - start_x) + (path_z[max_lateral_dev_idx] - start_z) * (end_z - start_z)) / \
            ((end_x - start_x)**2 + (end_z - start_z)**2)
        t = max(0, min(1, t)) # Clamp t to be on the segment
        proj_x = start_x + t * (end_x - start_x)
        proj_z = start_z + t * (end_z - start_z)
        ax_xz.annotate("", xy=(path_x[max_lateral_dev_idx], path_z[max_lateral_dev_idx]), xytext=(proj_x, proj_z),
                         arrowprops=dict(arrowstyle="<->", color='blue', lw=1.5), zorder=2)


    ax_xz.set_xlabel("Side Movement (X, meters)")
    ax_xz.set_ylabel("Forward Movement (Z, meters)")
    ax_xz.set_title("Top-Down View (Path from Start to End)")
    ax_xz.legend(loc='best')
    ax_xz.grid(True)
    ax_xz.axis('equal')

    # --- Side View (Plotting Z vs Y) ---
    ax_zy = axs[1]
    ax_zy.plot(path_z, path_y, marker='.', linestyle='-', label="Your Path", zorder=2) # Z on X-axis for intuitive "forward"
    ax_zy.plot([start_z, end_z], [start_y, end_y], 'k--', label="Direct Path", zorder=1) # Ideal direct path
    ax_zy.plot(start_z, start_y, 'go', markersize=10, label="Start", zorder=3)
    ax_zy.plot(end_z, end_y, 'ro', markersize=10, label="End", zorder=3)

    # Find and show max vertical deviation
    max_vert_dev_val = 0
    max_vert_dev_idx = -1
    # Vector P1P2
    P1 = np.array([start_z, start_y])
    P2 = np.array([end_z, end_y])
    if not np.allclose(P1, P2):
        for idx in range(len(path_z)):
            P0 = np.array([path_z[idx], path_y[idx]])
            # Perpendicular distance from P0 to line P1P2
            dist = np.abs(np.cross(P2-P1, P0-P1)) / np.linalg.norm(P2-P1)
            if dist > max_vert_dev_val:
                max_vert_dev_val = dist
                max_vert_dev_idx = idx
    
    if max_vert_dev_idx != -1 and max_vert_dev_val > 0.01: # Only show if deviation > 1cm
        ax_zy.plot(path_z[max_vert_dev_idx], path_y[max_vert_dev_idx], 'mo', markersize=7, label=f"Max Vertical Dev ({max_vert_dev_val*100:.1f}cm)", zorder=3)
        # Arrow from ideal line to point of max deviation
        t = np.dot(np.array([path_z[max_vert_dev_idx], path_y[max_vert_dev_idx]]) - P1, P2 - P1) / np.dot(P2 - P1, P2 - P1)
        t = max(0, min(1, t))
        proj_on_line = P1 + t * (P2 - P1)
        ax_zy.annotate("", xy=(path_z[max_vert_dev_idx], path_y[max_vert_dev_idx]), xytext=(proj_on_line[0], proj_on_line[1]),
                         arrowprops=dict(arrowstyle="<->", color='magenta', lw=1.5), zorder=2)


    ax_zy.set_xlabel("Forward Movement (Z, meters)")
    ax_zy.set_ylabel("Upward Movement (Y, meters)")
    ax_zy.set_title("Side View (Path from Start to End)")
    ax_zy.legend(loc='best')
    ax_zy.grid(True)
    ax_zy.axis('equal')

    plt.tight_layout(rect=[0, 0.05, 1, 0.90]) 
    
    # Add a text box with improvement tips based on curve_type
    tips = ""
    if curve_type == "push":
        tips = "Tip: 'Push' error detected. Try to avoid arcing the pistol upwards too much.\nAim for a more direct path to the target after the initial lift."
    elif curve_type == "swing":
        tips = "Tip: 'Swing' error detected. Reduce side-to-side or excessive vertical scooping.\nFocus on a controlled, direct movement towards the target."
    elif curve_type == "other" and motion_length_cm > (euclidean(actual_start_coords, actual_end_coords)*100 * 1.2): # If path is 20% longer than direct
        tips = "Tip: Path seems inefficient. Try for a smoother, more direct movement."
    elif curve_type == "line":
        tips = "Good job! Path is quite direct."
        
    if tips:
        fig.text(0.5, 0.01, tips, ha='center', va='bottom', fontsize=10, wrap=True, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    safe_drill_uid = "".join(c if c.isalnum() or c in ('-', '_', '.') else "_" for c in drill_uid)
    plot_filename = os.path.join(output_dir, f"user_drill_{safe_drill_uid}_presentation.png")
    try:
        plt.savefig(plot_filename)
        print(f"  User plot saved: {plot_filename}")
    except Exception as e:
        print(f"  Error saving user plot {plot_filename}: {e}")
    plt.close(fig)