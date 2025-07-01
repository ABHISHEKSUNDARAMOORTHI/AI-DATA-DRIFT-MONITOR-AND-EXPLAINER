import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, chi2_contingency
import logging

# Import the utility function for converting numpy types
# Ensure utils.py exists and has convert_numpy_types
from utils import convert_numpy_types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_psi(baseline_series, current_series, num_bins=10):
    """
    Calculates the Population Stability Index (PSI) for numerical features.
    Handles NaN values by placing them in a separate bin.
    """
    # Drop NaNs for initial binning, but count them for separate bin
    baseline_clean = baseline_series.dropna()
    current_clean = current_series.dropna()

    baseline_nan_count = baseline_series.isna().sum()
    current_nan_count = current_series.isna().sum()

    if baseline_clean.empty and current_clean.empty:
        return 0.0 # No data to compare

    # Determine common bins
    all_data = pd.concat([baseline_clean, current_clean]).unique()
    if len(all_data) < num_bins: # For small number of unique values, use unique values as boundaries
        bins = np.sort(all_data)
    else: # For more data, create quantile-based bins from baseline data
        try:
            # Create bins based on the baseline distribution
            bins = np.percentile(baseline_clean, np.linspace(0, 100, num_bins + 1))
            bins[0] = -np.inf # Ensure first bin captures all low values
            bins[-1] = np.inf # Ensure last bin captures all high values
        except IndexError: # Handle cases where baseline_clean might be too small after dropna
            return 0.0 # Not enough data for meaningful bins

    # Calculate counts in each bin
    baseline_counts, _ = np.histogram(baseline_clean, bins=bins)
    current_counts, _ = np.histogram(current_clean, bins=bins)

    # Add NaN counts as a separate "bin"
    baseline_counts = np.append(baseline_counts, baseline_nan_count)
    current_counts = np.append(current_counts, current_nan_count)

    # Convert counts to percentages, adding a small epsilon to avoid division by zero
    baseline_pct = baseline_counts / (baseline_counts.sum() + 1e-10)
    current_pct = current_counts / (current_counts.sum() + 1e-10)

    # Avoid log(0) by adding a small epsilon
    psi_values = []
    for i in range(len(baseline_pct)):
        # Calculate PSI only if both percentages are non-zero or if one is zero and the other is not significantly large
        # If both are 0, contribution is 0.
        if baseline_pct[i] == 0 and current_pct[i] == 0:
            psi_values.append(0.0)
        else:
            # Add epsilon to prevent log(0) and division by zero
            psi_values.append((current_pct[i] - baseline_pct[i]) * np.log((current_pct[i] + 1e-10) / (baseline_pct[i] + 1e-10)))
    
    return np.sum(psi_values)

def calculate_statistical_drift(baseline_series, current_series, column_type):
    """
    Calculates appropriate statistical drift metric based on column type.
    """
    if baseline_series.empty and current_series.empty:
        return 0.0 # No data to compare

    if column_type == 'numerical':
        # Wasserstein distance (Earth Mover's Distance) for numerical distributions
        # Robust to different number of samples and non-overlapping ranges
        # Normalize data to a common scale if distributions can be very different
        min_val = min(baseline_series.min(), current_series.min())
        max_val = max(baseline_series.max(), current_series.max())
        
        # Avoid division by zero if min_val == max_val
        if max_val - min_val == 0:
            return 0.0
        
        baseline_normalized = (baseline_series - min_val) / (max_val - min_val)
        current_normalized = (current_series - min_val) / (max_val - min_val)
        
        return wasserstein_distance(baseline_normalized.dropna(), current_normalized.dropna())

    elif column_type == 'categorical':
        # Chi-squared test for categorical distributions
        # Null hypothesis: the two distributions are the same. Low p-value suggests drift.
        # Create a combined frequency table
        baseline_counts = baseline_series.value_counts(normalize=False) # Use raw counts for chi2
        current_counts = current_series.value_counts(normalize=False)

        # Get all unique categories across both series
        all_categories = pd.Index(list(baseline_counts.index) + list(current_counts.index)).unique()

        # Reindex both series to have all categories, filling missing with 0
        baseline_aligned = baseline_counts.reindex(all_categories, fill_value=0)
        current_aligned = current_counts.reindex(all_categories, fill_value=0)

        # Create a contingency table
        observed = pd.DataFrame({
            'baseline': baseline_aligned,
            'current': current_aligned
        })
        
        # Remove categories where both counts are zero to avoid errors in chi2
        observed = observed[(observed['baseline'] > 0) | (observed['current'] > 0)]

        if observed.empty:
            return 0.0
        
        # Ensure that the observed table has at least 2 rows and 2 columns for chi2_contingency
        if observed.shape[0] < 2 or observed.shape[1] < 2:
            # Fallback for simple cases (e.g., only one category)
            # If distributions are exactly the same, no drift. Otherwise, some drift.
            if (baseline_aligned == current_aligned).all():
                return 0.0
            else:
                return 0.5 # Indicate some drift if counts differ for simple cases

        # Perform Chi-squared test
        try:
            chi2, p_value, _, _ = chi2_contingency(observed)
            # A common way to convert p-value to a "drift score": 1 - p_value
            # This means a very low p-value (strong evidence of difference) gives a high drift score.
            return 1 - p_value
        except ValueError as e:
            logging.warning(f"Chi-squared test failed for column (categorical): {e}. Falling back to mode comparison.")
            # This can happen if, for example, expected frequencies are too small.
            # In such cases, a simple difference in most common categories might be indicative.
            if not baseline_series.empty and not current_series.empty:
                top_baseline = baseline_series.mode()[0] if not baseline_series.mode().empty else None
                top_current = current_series.mode()[0] if not current_series.mode().empty else None
                if top_baseline != top_current:
                    return 0.5 # Indicate some drift if modes differ
            return 0.0 # Unable to calculate, assume no significant drift

    return 0.0 # Default for unknown types or types that can't be compared

def analyze_drift(baseline_df, current_df, psi_threshold=0.25, chi2_threshold=0.05):
    """
    Analyzes data drift between baseline and current dataframes.
    Returns a comprehensive drift report.
    """
    drift_report = {
        'schema_drift': {
            'added_columns': [],
            'removed_columns': [],
            'changed_columns': {}
        },
        'column_drift': {}
    }

    baseline_cols = set(baseline_df.columns)
    current_cols = set(current_df.columns)

    # --- Schema Drift ---
    drift_report['schema_drift']['added_columns'] = list(current_cols - baseline_cols)
    drift_report['schema_drift']['removed_columns'] = list(baseline_cols - current_cols)

    common_columns = list(baseline_cols.intersection(current_cols))

    for col in common_columns:
        baseline_dtype = baseline_df[col].dtype
        current_dtype = current_df[col].dtype
        
        col_drift_details = {}
        col_drift_details['drift_detected'] = False # Flag to indicate if any significant drift is found for this column
        
        # --- Type Change Detection (part of schema drift, but also impacts column drift details) ---
        if str(baseline_dtype) != str(current_dtype):
            drift_report['schema_drift']['changed_columns'][col] = {
                'old_type': str(baseline_dtype),
                'new_type': str(current_dtype)
            }
            col_drift_details['data_type_changed'] = True
            col_drift_details['drift_detected'] = True
        else:
            col_drift_details['data_type_changed'] = False
        
        col_drift_details['type_old'] = str(baseline_dtype)
        col_drift_details['type_new'] = str(current_dtype)

        # --- Missing Values Drift ---
        null_pct_old = baseline_df[col].isnull().sum() / len(baseline_df) * 100 if len(baseline_df) > 0 else 0.0
        null_pct_new = current_df[col].isnull().sum() / len(current_df) * 100 if len(current_df) > 0 else 0.0
        
        col_drift_details['null_pct_old'] = float(null_pct_old)
        col_drift_details['null_pct_new'] = float(null_pct_new)
        
        # Define a threshold for "missing values drift"
        if abs(null_pct_new - null_pct_old) > 5: # Arbitrary threshold, adjust as needed
            col_drift_details['missing_values_drift'] = True
            col_drift_details['drift_detected'] = True
        else:
            col_drift_details['missing_values_drift'] = False


        # --- Data Type Check for Statistical Drift ---
        is_numeric_baseline = pd.api.types.is_numeric_dtype(baseline_df[col])
        is_numeric_current = pd.api.types.is_numeric_dtype(current_df[col])

        # Infer common column type for drift calculation (numerical or categorical)
        column_type_for_stats = None
        current_col_drift_score = 0.0

        if is_numeric_baseline and is_numeric_current:
            column_type_for_stats = 'numerical'
            col_drift_details['mean_old'] = float(baseline_df[col].mean())
            col_drift_details['mean_new'] = float(current_df[col].mean())
            col_drift_details['std_old'] = float(baseline_df[col].std())
            col_drift_details['std_new'] = float(current_df[col].std())
            
            # Calculate PSI for numerical
            psi_score = calculate_psi(baseline_df[col], current_df[col])
            col_drift_details['psi_score'] = float(psi_score)
            current_col_drift_score = psi_score
            
        elif not is_numeric_baseline and not is_numeric_current: # Both are non-numeric (categorical/object)
            column_type_for_stats = 'categorical'

            # Categorical Drift Details
            vc_baseline = baseline_df[col].value_counts(normalize=True) # Proportions
            vc_current = current_df[col].value_counts(normalize=True)
            
            # Convert proportions to percentage for display, handling cases where value_counts might be empty
            top_baseline_dict = (vc_baseline.nlargest(5) * 100).round(2).to_dict() if not vc_baseline.empty else {}
            top_current_dict = (vc_current.nlargest(5) * 100).round(2).to_dict() if not vc_current.empty else {}

            new_categories = list(set(current_df[col].dropna().unique()) - set(baseline_df[col].dropna().unique()))
            missing_categories = list(set(baseline_df[col].dropna().unique()) - set(current_df[col].dropna().unique()))
            
            col_drift_details['category_drift'] = {
                'top_categories_old': convert_numpy_types(top_baseline_dict),
                'top_categories_new': convert_numpy_types(top_current_dict),
                'new_categories': new_categories,
                'missing_categories': missing_categories
            }
            if new_categories or missing_categories:
                col_drift_details['drift_detected'] = True # Categorical drift detected by new/missing categories
            
            # Calculate Chi-squared based drift score
            chi2_drift_score = calculate_statistical_drift(baseline_df[col], current_df[col], column_type_for_stats)
            col_drift_details['chi2_drift_score'] = float(chi2_drift_score)
            current_col_drift_score = chi2_drift_score

        else: # Type mismatch (e.g., numerical in old, categorical in new)
            column_type_for_stats = 'mixed'
            current_col_drift_score = 1.0 # High drift due to type change
            col_drift_details['drift_detected'] = True
        
        # Assign the calculated drift score to the main 'drift_score' key
        col_drift_details['drift_score'] = float(current_col_drift_score)
        
        # If no drift detected yet, check against a general drift threshold
        if not col_drift_details['drift_detected'] and current_col_drift_score > 0.1: # A general threshold for statistical drift
            col_drift_details['drift_detected'] = True

        drift_report['column_drift'][col] = col_drift_details

    logging.info("Drift analysis completed.")
    return drift_report

def generate_drift_summary_table(drift_report):
    """
    Generates a pandas DataFrame summary from the drift report for display.
    Includes safe formatting to avoid TypeError with None values.
    """
    summary_data = []
    # Ensure 'column_drift' key exists and is a dictionary
    column_drift_data = drift_report.get('column_drift', {})

    for col, details in column_drift_data.items():
        # Safely get drift_score, defaulting to 0.0 if not present or None
        drift_score = details.get('drift_score', 0.0)

        # Build comments string safely
        comments = 'N/A' # Default comment

        if 'mean_old' in details or 'mean_new' in details:
            # Safely get mean values, defaulting to 0 if None, before formatting
            mean_old_val = details.get('mean_old', 0)
            mean_new_val = details.get('mean_new', 0)
            comments = f"Mean: {mean_old_val:.2f} -> {mean_new_val:.2f}"
        elif details.get('category_drift', {}).get('new_categories'):
            new_cats = details['category_drift']['new_categories']
            # Ensure new_cats is a list and handle cases where it might be empty
            if isinstance(new_cats, list) and new_cats:
                comments = f"New categories: {', '.join(map(str, new_cats))}"
            else:
                comments = "New categories detected (details N/A)"
        elif details.get('category_drift', {}).get('missing_categories'):
            missing_cats = details['category_drift']['missing_categories']
            # Ensure missing_cats is a list and handle cases where it might be empty
            if isinstance(missing_cats, list) and missing_cats:
                comments = f"Missing categories: {', '.join(map(str, missing_cats))}"
            else:
                comments = "Missing categories detected (details N/A)"
        
        summary_data.append({
            'Column': col,
            'Drift Score': f"{drift_score:.2f}", # Format the safely retrieved score
            'Old Type': details.get('type_old', 'N/A'),
            'New Type': details.get('type_new', 'N/A'),
            'Missing % (Old)': f"{details.get('null_pct_old', 0.0):.2f}%", # Safe format
            'Missing % (New)': f"{details.get('null_pct_new', 0.0):.2f}%", # Safe format
            'Missing Drift': 'Yes' if details.get('missing_values_drift') else 'No',
            'Type Changed': 'Yes' if details.get('data_type_changed') else 'No', # Use 'data_type_changed' flag
            'Comments': comments
        })
    df_summary = pd.DataFrame(summary_data)
    if not df_summary.empty:
        # Convert 'Drift Score' column to numeric *after* formatting, for sorting
        df_summary['Drift Score Num'] = pd.to_numeric(df_summary['Drift Score'], errors='coerce')
        df_summary = df_summary.sort_values(by='Drift Score Num', ascending=False).drop(columns=['Drift Score Num'])
    return df_summary


def prepare_drift_summary_for_gemini(drift_report):
    """
    Prepares a concise summary of the drift report optimized for Gemini's input.
    Focuses on key changes and removes verbose details not needed for a high-level explanation.
    Ensures all numerical types are standard Python types for JSON serialization.
    """
    summary_for_ai = {
        "schema_drift_summary": drift_report.get('schema_drift', {}), # Use 'schema_drift' key
        "column_drift_summary": {}
    }

    # Process column drift to make it more concise and AI-friendly
    # Use 'column_drift' key from the main drift_report
    for col, details in drift_report.get('column_drift', {}).items():
        # Only include columns with a drift score or significant changes
        if details.get('drift_score', 0) > 0.1 or \
           details.get('data_type_changed') or \
           details.get('missing_values_drift') or \
           (details.get('category_drift', {}).get('new_categories') and len(details['category_drift']['new_categories']) > 0) or \
           (details.get('category_drift', {}).get('missing_categories') and len(details['category_drift']['missing_categories']) > 0):

            col_summary = {
                "drift_score": details.get('drift_score'),
                "data_type_old": details.get('type_old'),
                "data_type_new": details.get('type_new'),
                "missing_values_drift_detected": details.get('missing_values_drift'),
                "null_pct_old": details.get('null_pct_old'),
                "null_pct_new": details.get('null_pct_new'),
                "statistics_drift": {}, # Store relevant numerical/categorical stats
            }

            # Add numerical specific stats if applicable
            if 'mean_old' in details:
                col_summary['statistics_drift']['mean_old'] = details.get('mean_old')
                col_summary['statistics_drift']['mean_new'] = details.get('mean_new')
                col_summary['statistics_drift']['std_old'] = details.get('std_old')
                col_summary['statistics_drift']['std_new'] = details.get('std_new')

            # Add categorical specific stats if applicable
            if 'category_drift' in details:
                cat_drift = details['category_drift']
                col_summary['statistics_drift']['top_categories_old'] = cat_drift.get('top_categories_old')
                col_summary['statistics_drift']['top_categories_new'] = cat_drift.get('top_categories_new')
                col_summary['statistics_drift']['new_categories'] = cat_drift.get('new_categories')
                col_summary['statistics_drift']['missing_categories'] = cat_drift.get('missing_categories')

            summary_for_ai['column_drift_summary'][col] = col_summary
    
    # Crucial step: Convert NumPy types to Python native types for JSON serialization
    return convert_numpy_types(summary_for_ai)

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    print("Running drift_analysis.py example...")
    
    # Sample DataFrames
    data_old = {
        'customer_id': [1, 2, 3, 4, 5],
        'customer_age': [25, 30, 35, 40, 45],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'income': [50000, 60000, 55000, 70000, 65000],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA'],
        'purchases_last_month': [2, 3, 1, 4, 2],
        'product_views': [10, 12, 8, 15, 11] # Adding a column to baseline to simulate removal
    }
    df_baseline = pd.DataFrame(data_old)

    data_new = {
        'customer_id': [6, 7, 8, 9, 10],
        'customer_age': [30, 38, 45, 50, 55], # Age drift
        'gender': ['Female', 'Female', 'Male', 'Female', 'Female'], # Gender drift
        'income': [52000, 63000, 58000, 72000, 68000], # Slight income drift
        'city': ['NY', 'Houston', 'NY', 'SF', 'LA'], # New city 'Houston'
        'product_reviews': [5, 8, 4, 6, 7], # New column 'product_reviews'
        'purchases_last_month': [1, 2, 0, 3, 1] # Drift in purchases
    }
    df_current = pd.DataFrame(data_new)
    # Simulate missing values for null_pct_drift test
    df_current.loc[0:10, 'customer_age'] = np.nan 

    print("\n--- Performing Drift Analysis ---")
    report = analyze_drift(df_baseline, df_current)
    # print(json.dumps(report, indent=2)) # This might still error without convert_numpy_types here too

    print("\n--- Generating Summary Table ---")
    summary_df = generate_drift_summary_table(report)
    print(summary_df)

    print("\n--- Preparing Summary for Gemini ---")
    gemini_summary = prepare_drift_summary_for_gemini(report)
    import json # Import json for testing print
    print("Gemini Summary (first 500 chars):")
    print(json.dumps(gemini_summary, indent=2)[:500]) # Ensure it's serializable now

    print("\nDrift Analysis Example Complete.")