# Wepan_presentation_analysis
Home assignment including data exploration, feature engineering and visualizations.
the workflow processes raw sensor data, identifies specific drills, analyzes the movement patterns during the presentation phase, calculates relevant performance metrics, classifies the movement type, and visualizes the results for both technical analysis and user feedback.
## Main workflow:
1. Data Loading and Parsing : parse_event_file function is used to read raw data from a .event file.
2. Drill Structuring: A dictionary (drills_for_analysis) is built to store key information for each drill, such as the start and end timestamps.
3. Identifying Presentation Trajectory and Key Timestamps:For each identified drill, the workflow aims to find the specific portion of the trajectory data  that corresponds to the weapon "presentation".
4. Calculating Presentation Metrics:  various metrics are calculated: speed, motion length, reaction time, curve shape.
    curve shape of the trajectory is classified using either classify_presentation_curve (deviation-based) or classify_presentation_curve_polynomial_fit (fit-based).
5. Visualization, Reporting and exploratory data analysis.
   
## Functions:
### parse_event_file(file_path):
Reads a .event file.
Parses header definitions and event data.
Handles dynamic headers and potential mismatches in field counts.
Returns header definitions and a list of parsed event dictionaries.

### plot_drill_presentation(scenario_id, drill_uid, trajectory_df, T_pres_start, T_pres_end, weapon_pos_cols, curve_type, curve_fit_polynomial, base_output_dir="plots"):
Generates and saves a 2D plot of the weapon's trajectory during a drill presentation.
Shows top-down (XZ) and side (ZY) views.
Marks the start and end points of the presentation.
Includes drill information in the plot title.
Saves the plot as a PNG file in a scenario-specific directory.

### plot_drill_presentation_user_focused(scenario_id, drill_uid, trajectory_df, T_pres_start, T_pres_end, weapon_pos_cols, curve_type, speed_ms, motion_length_cm, reaction_time_ms, base_output_dir="plots_user"):
Creates and saves a user-focused plot of the weapon presentation path.
Displays the user's path and the ideal direct path in XZ and ZY views.
Highlights the start and end points.
Shows max lateral and vertical deviations with annotations.
Includes key metrics (curve type, speed, reaction time, motion length) in the title.
Adds improvement tips based on the curve type.
Saves the plot as a PNG file in a scenario-specific user-focused directory.

### classify_presentation_curve(trajectory_df, weapon_pos_cols, motion_length_m, direct_dist_m, curve_thresholds):
Analyzes a weapon trajectory to classify its curve shape ("line", "push", "swing", or "other").
Uses geometric analysis based on path length, direct distance, and deviations from the ideal line in vertical (Y) and lateral (X) dimensions.
Compares calculated deviations against provided thresholds to make the classification.

### fit_and_evaluate_polynomial(x_coords, y_coords, degree):
A helper function to fit a polynomial of a given degree to a set of x and y coordinates.
Calculates and returns the polynomial coefficients and the R-squared value, which indicates how well the polynomial fits the data.

### classify_presentation_curve_polynomial_fit(trajectory_df, weapon_pos_cols, curve_thresholds):
Classifies the curve shape ("line", "push", "swing", or "other") using a different method based on polynomial fitting.
Fits linear and parabolic models to the trajectory data in the YZ (side) and XZ (top-down) planes.
Uses the R-squared values and the coefficients of the parabolic fit to determine if the path is more like a line or an arc (push/swing).
Includes a log of the decision process for clarity.
