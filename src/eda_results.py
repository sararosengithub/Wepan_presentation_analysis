import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_exploratory_analysis(results_df):
    """
    Performs exploratory data analysis on the aggregated drill presentation metrics.
    Args:
        results_df (pd.DataFrame): DataFrame containing metrics from all scenarios and drills.
    """
    print("\n\n--- Exploratory Data Analysis ---")

    # Filter out rows where analysis might have failed or metrics are missing
    # We'll focus on successful analyses for these summary stats.
    valid_results = results_df.dropna(subset=['speed_ms', 'motion_length_cm', 'curve_type']).copy()
    valid_results = valid_results[valid_results['error'].isnull()] # Only where no error reported

    if valid_results.empty:
        print("No valid results available for EDA after filtering errors/NaNs.")
        return

    # --- 1. Distribution of Presentation Speed ---
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_results['speed_ms'], kde=True, bins=20)
    plt.title('Distribution of Presentation Speed (ms)')
    plt.xlabel('Speed (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eda_plot_speed_distribution.png")
    plt.show()

    # --- 2. Distribution of Motion Length ---
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_results['motion_length_cm'], kde=True, bins=20)
    plt.title('Distribution of Presentation Motion Length (cm)')
    plt.xlabel('Motion Length (cm)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eda_plot_motion_distribution.png")
    plt.show()

    # --- 3. Frequency of Curve Types ---
    if 'curve_type' in valid_results.columns and not valid_results['curve_type'].empty:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='curve_type', data=valid_results, order=valid_results['curve_type'].value_counts().index)
        plt.title('Frequency of Presentation Curve Types')
        plt.xlabel('Curve Type')
        plt.ylabel('Count')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig("eda_plot_curve_type_frequency.png")
        plt.show()
    else:
        print("Skipping curve type frequency plot: 'curve_type' column missing or empty.")


    # --- 4. Speed vs. Motion Length ---
    if 'curve_type' in valid_results.columns:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x='speed_ms', y='motion_length_cm', hue='curve_type', data=valid_results, alpha=0.7, s=50)
        plt.title('Presentation Speed vs. Motion Length')
        plt.xlabel('Speed (ms)')
        plt.ylabel('Motion Length (cm)')
        plt.legend(title='Curve Type')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("eda_plot_speed_vs_motion.png")
        plt.show()
    else: # Fallback if no curve_type
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x='speed_ms', y='motion_length_cm', data=valid_results, alpha=0.7, s=50)
        plt.title('Presentation Speed vs. Motion Length')
        plt.xlabel('Speed (ms)')
        plt.ylabel('Motion Length (cm)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("eda_plot_speed_vs_motion_no_hue.png")
        plt.show()


    # --- 5. Comparison of Metrics Across Scenarios ---
    if 'scenario_id' in valid_results.columns and valid_results['scenario_id'].nunique() > 1:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x='scenario_id', y='speed_ms', data=valid_results)
        plt.title('Presentation Speed by Scenario')
        plt.xlabel('Scenario ID')
        plt.ylabel('Speed (ms)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        sns.boxplot(x='scenario_id', y='motion_length_cm', data=valid_results)
        plt.title('Motion Length by Scenario')
        plt.xlabel('Scenario ID')
        plt.ylabel('Motion Length (cm)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("eda_plot_metrics_by_scenario_boxplot.png")
        plt.show()

        if 'curve_type' in valid_results.columns:
            plt.figure(figsize=(12, 7))
            # Create a pivot table for stacked bar chart
            scenario_curve_counts = valid_results.groupby(['scenario_id', 'curve_type']).size().unstack(fill_value=0)
            if not scenario_curve_counts.empty:
                scenario_curve_counts.plot(kind='bar', stacked=True)
                plt.title('Curve Types by Scenario (Stacked)')
                plt.xlabel('Scenario ID')
                plt.ylabel('Count')
                plt.legend(title='Curve Type')
                plt.xticks(rotation=45)
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig("eda_plot_curve_types_by_scenario_stacked.png")
                plt.show()
            else:
                print("Not enough data to plot curve types by scenario (stacked).")

    elif 'scenario_id' in valid_results.columns:
        print("Only one scenario ID found. Skipping scenario comparison plots.")
    else:
        print("Skipping scenario comparison plots: 'scenario_id' column missing.")


    print("EDA plots generated and saved (if any).")

