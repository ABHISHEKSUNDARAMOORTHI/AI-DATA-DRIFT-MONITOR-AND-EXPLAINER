import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_drift_heatmap(drift_report):
    """
    Generates a heatmap of drift scores for all columns.
    """
    if not drift_report or 'column_drift' not in drift_report:
        logging.warning("No drift report or column_drift data to generate heatmap.")
        return None

    scores = {col: details.get('drift_score', 0.0)
              for col, details in drift_report['column_drift'].items()}

    if not scores:
        logging.info("No columns with drift scores found for heatmap.")
        return None

    # Create a DataFrame for the heatmap
    df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Drift Score'])
    df_scores = df_scores.sort_values(by='Drift Score', ascending=False)

    if df_scores.empty:
        logging.info("Drift scores DataFrame is empty for heatmap.")
        return None

    # Use matplotlib for the heatmap
    fig, ax = plt.subplots(figsize=(10, len(df_scores) * 0.5 + 2)) # Adjust figure size dynamically
    
    # Define a custom colormap that goes from green (low drift) to red (high drift)
    cmap = sns.diverging_palette(145, 10, as_cmap=True, s=90, l=40, sep=1, center="dark")
    
    sns.heatmap(df_scores, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, ax=ax,
                cbar_kws={'label': 'Drift Score'})

    ax.set_title('Column Drift Heatmap', fontsize=16)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Column', fontsize=12)
    plt.tight_layout()
    return fig

def plot_column_comparison(baseline_df, current_df, column):
    """
    Generates a Plotly chart for comparing the distribution of a single column.
    Handles numerical and categorical columns.
    """
    if column not in baseline_df.columns or column not in current_df.columns:
        logging.error(f"Column '{column}' not found in both dataframes for comparison plot.")
        return None

    # Determine if the column is numerical or categorical
    is_numerical = pd.api.types.is_numeric_dtype(baseline_df[column]) and \
                   pd.api.types.is_numeric_dtype(current_df[column])

    try:
        if is_numerical:
            # Drop NaNs for plotting, as Plotly handles them by default but can make visualization cleaner
            baseline_data = baseline_df[column].dropna()
            current_data = current_df[column].dropna()

            # Concatenate data for plotting
            plot_df = pd.DataFrame({
                'Value': pd.concat([baseline_data, current_data]),
                'Dataset': ['Baseline'] * len(baseline_data) + ['Current'] * len(current_data)
            })
            
            if plot_df.empty:
                logging.warning(f"No valid numerical data for column '{column}' to plot.")
                return None

            # Histogram for numerical distribution comparison
            fig = px.histogram(plot_df, x='Value', color='Dataset',
                               marginal='box', # Show box plot for distribution summary
                               barmode='overlay', # Overlay histograms
                               histnorm='probability density', # Normalize to probability density for comparison
                               title=f'Distribution Comparison for: {column}',
                               opacity=0.6,
                               template="plotly_dark", # Use a dark theme
                               color_discrete_map={'Baseline': '#4F46E5', 'Current': '#22C55E'} # Custom colors
                            )
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Probability Density",
                legend_title="Dataset",
                hovermode="x unified"
            )
            return fig

        else: # Categorical or object type
            # Get value counts for both datasets
            baseline_counts = baseline_df[column].value_counts(normalize=True).reset_index()
            baseline_counts.columns = ['Category', 'Percentage']
            baseline_counts['Dataset'] = 'Baseline'

            current_counts = current_df[column].value_counts(normalize=True).reset_index()
            current_counts.columns = ['Category', 'Percentage']
            current_counts['Dataset'] = 'Current'

            plot_df = pd.concat([baseline_counts, current_counts])
            
            if plot_df.empty:
                logging.warning(f"No valid categorical data for column '{column}' to plot.")
                return None

            # Sort categories by baseline percentage to keep order consistent
            order = baseline_counts.sort_values('Percentage', ascending=False)['Category'].tolist()
            
            # Bar chart for categorical distribution comparison
            fig = px.bar(plot_df, x='Category', y='Percentage', color='Dataset',
                         barmode='group',
                         title=f'Categorical Distribution Comparison for: {column}',
                         template="plotly_dark",
                         color_discrete_map={'Baseline': '#4F46E5', 'Current': '#22C55E'},
                         category_orders={"Category": order} # Apply order
                        )
            fig.update_layout(
                xaxis_title="Category",
                yaxis_title="Percentage",
                legend_title="Dataset",
                hovermode="x unified"
            )
            fig.update_yaxes(tickformat=".1%") # Format y-axis as percentage
            return fig

    except Exception as e:
        logging.error(f"Error generating Plotly comparison for column '{column}': {e}", exc_info=True)
        return None


def plot_target_drift(baseline_df, current_df, target_column):
    """
    Plots the distribution of a specified target column for baseline vs current data.
    Uses matplotlib and seaborn.
    """
    if target_column not in baseline_df.columns or target_column not in current_df.columns:
        logging.error(f"Target column '{target_column}' not found in both dataframes for target drift plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if the target column is numerical
    is_numerical = pd.api.types.is_numeric_dtype(baseline_df[target_column]) and \
                   pd.api.types.is_numeric_dtype(current_df[target_column])

    try:
        if is_numerical:
            sns.histplot(baseline_df[target_column].dropna(), color='#4F46E5', alpha=0.6, stat='density', kde=True, label='Baseline', ax=ax)
            sns.histplot(current_df[target_column].dropna(), color='#22C55E', alpha=0.6, stat='density', kde=True, label='Current', ax=ax)
            ax.set_title(f'Distribution of Target Variable: {target_column}', fontsize=16)
            ax.set_ylabel('Density', fontsize=12)
        else: # Assume categorical for non-numerical
            # Combine and normalize value counts
            baseline_counts = baseline_df[target_column].value_counts(normalize=True).sort_index()
            current_counts = current_df[target_column].value_counts(normalize=True).sort_index()

            # Create a combined index of all unique categories
            all_categories = pd.Index(list(baseline_counts.index) + list(current_counts.index)).unique().sort_values()

            # Reindex to ensure all categories are present, filling missing with 0
            baseline_aligned = baseline_counts.reindex(all_categories, fill_value=0)
            current_aligned = current_counts.reindex(all_categories, fill_value=0)

            # Bar plot for categorical target
            width = 0.35
            x = np.arange(len(all_categories))
            ax.bar(x - width/2, baseline_aligned, width, label='Baseline', color='#4F46E5', alpha=0.8)
            ax.bar(x + width/2, current_aligned, width, label='Current', color='#22C55E', alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            ax.set_title(f'Distribution of Categorical Target: {target_column}', fontsize=16)
            ax.set_ylabel('Proportion', fontsize=12)
            plt.tight_layout() # Adjust layout to prevent labels overlapping

        ax.set_xlabel(target_column, fontsize=12)
        ax.legend(fontsize=10)
        
        # Apply dark theme
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#0F172A')
        ax.tick_params(colors='#F8FAFC')
        ax.xaxis.label.set_color('#F8FAFC')
        ax.yaxis.label.set_color('#F8FAFC')
        ax.title.set_color('#E2E8F0')
        ax.spines['top'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')
        ax.spines['left'].set_color('#475569')
        ax.spines['right'].set_color('#475569')
        ax.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC')

        return fig

    except Exception as e:
        logging.error(f"Error generating target drift plot for column '{target_column}': {e}", exc_info=True)
        return None

# Example Usage (for direct testing of this file)
if __name__ == '__main__':
    print("Running visualizer.py example...")

    # Sample DataFrames
    data_old = {
        'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'customer_age': [25, 30, 35, 40, 45, 28, 33, 38, 42, 48],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'income': [50000, 60000, 55000, 70000, 65000, 52000, 63000, 58000, 72000, 68000],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA', 'NY', 'LA', 'SF', 'NY', 'LA'],
        'purchases_last_month': [2, 3, 1, 4, 2, 2, 3, 1, 4, 2],
        'target_binary': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], # Numerical target
        'target_category': ['A', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'B', 'A'] # Categorical target
    }
    df_baseline = pd.DataFrame(data_old)

    data_new = {
        'customer_id': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'customer_age': [30, 38, 45, 50, 55, 35, 40, 48, 52, 58], # Age drift
        'gender': ['Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Male'], # Gender drift
        'income': [52000, 63000, 58000, 72000, 68000, 55000, 66000, 61000, 75000, 71000], # Slight income drift
        'city': ['NY', 'Houston', 'NY', 'SF', 'LA', 'Austin', 'NY', 'SF', 'Houston', 'LA'], # New cities
        'purchases_last_month': [1, 2, 0, 3, 1, 1, 2, 0, 3, 1], # Drift in purchases
        'target_binary': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1], # Numerical target drift
        'target_category': ['B', 'C', 'B', 'C', 'B', 'C', 'B', 'C', 'B', 'C'] # Categorical target drift
    }
    df_current = pd.DataFrame(data_new)

    # --- Test plot_column_comparison ---
    print("\n--- Testing plot_column_comparison (Numerical: customer_age) ---")
    fig_age = plot_column_comparison(df_baseline, df_current, 'customer_age')
    if fig_age:
        fig_age.show() # This will open the plot in your browser or an interactive window

    print("\n--- Testing plot_column_comparison (Categorical: gender) ---")
    fig_gender = plot_column_comparison(df_baseline, df_current, 'gender')
    if fig_gender:
        fig_gender.show()

    # --- Test plot_target_drift ---
    print("\n--- Testing plot_target_drift (Numerical: target_binary) ---")
    fig_target_binary = plot_target_drift(df_baseline, df_current, 'target_binary')
    if fig_target_binary:
        plt.show() # For matplotlib figures

    print("\n--- Testing plot_target_drift (Categorical: target_category) ---")
    fig_target_category = plot_target_drift(df_baseline, df_current, 'target_category')
    if fig_target_category:
        plt.show()

    # --- Test plot_drift_heatmap (requires a mock drift_report) ---
    print("\n--- Testing plot_drift_heatmap ---")
    mock_drift_report = {
        'column_drift': {
            'customer_age': {'drift_score': 0.85},
            'gender': {'drift_score': 0.60},
            'income': {'drift_score': 0.35},
            'city': {'drift_score': 0.92},
            'purchases_last_month': {'drift_score': 0.70},
            'target_binary': {'drift_score': 0.55},
            'target_category': {'drift_score': 0.78},
            'customer_id': {'drift_score': 0.10} # Low drift
        }
    }
    heatmap_fig = plot_drift_heatmap(mock_drift_report)
    if heatmap_fig:
        plt.show()

    print("\nVisualizer Example Complete.")