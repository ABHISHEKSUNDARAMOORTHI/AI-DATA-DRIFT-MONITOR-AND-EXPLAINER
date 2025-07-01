import numpy as np
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_numpy_types(obj):
    """
    Recursively converts NumPy-specific types (like np.int64, np.float64, np.bool_)
    within a dictionary or list to standard Python types for JSON serialization.
    Updated to be compatible with NumPy 2.0 by using np.integer and np.floating.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    # Use np.integer for all integer types (replaces np.int_)
    elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # Use np.floating for all float types (replaces np.float_)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert NumPy arrays to Python lists
    else:
        return obj

def create_session_summary_json(baseline_filename, current_filename, drift_report, drift_summary_table, ai_explanation):
    """
    Creates a JSON-serializable dictionary summarizing the current session's analysis.
    This includes file names, the full drift report, the summary table, and the AI explanation.
    Ensures all NumPy types are converted to standard Python types.

    Args:
        baseline_filename (str): Name of the uploaded baseline file.
        current_filename (str): Name of the uploaded current file.
        drift_report (dict): The detailed drift analysis report.
        drift_summary_table (list or None): The drift summary table, expected as a list of dicts.
                                            Pass None if not available.
        ai_explanation (str): The AI-generated explanation.

    Returns:
        dict: A dictionary containing the session summary, ready for JSON serialization.
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "baseline_file": baseline_filename,
        "current_file": current_filename,
        "drift_report": drift_report,
        "drift_summary_table": drift_summary_table,
        "ai_explanation": ai_explanation
    }

    # Ensure all NumPy types within the complex drift_report are converted
    cleaned_summary = convert_numpy_types(summary)
    
    logging.info("Session summary JSON created and cleaned of NumPy types.")
    return cleaned_summary

# Example Usage (for direct testing of this file)
if __name__ == '__main__':
    print("Running utils.py example...")

    # --- Test convert_numpy_types ---
    print("\n--- Testing convert_numpy_types ---")
    # Using np.array for testing as specific dtypes are usually created this way
    test_data = {
        "int_val": np.array(123, dtype=np.int64),
        "float_val": np.array(45.67, dtype=np.float32),
        "bool_val": np.array(True, dtype=bool),
        "array_val": np.array([1, 2, 3], dtype=np.int64),
        "nested_list": [np.array(1.1, dtype=np.float64), np.array(2, dtype=np.int64)],
        "nested_dict": {"key1": np.array(100, dtype=np.int32), "key2": "string"},
        "standard_str": "hello",
        "standard_int": 789
    }

    cleaned_data = convert_numpy_types(test_data)
    print("Original data types (from test_data values):")
    for k, v in test_data.items():
        # For numpy scalars, type(v) will be np.int64 etc.
        # For non-numpy, it will be standard python types.
        print(f"  {k}: {type(v)}, Value: {v}")
    
    print("\nCleaned data types (after conversion):")
    for k, v in cleaned_data.items():
        # After conversion, these should be standard python types
        print(f"  {k}: {type(v)}, Value: {v}")
    
    # Verify JSON serializability
    try:
        json_output = json.dumps(cleaned_data, indent=2)
        print("\nCleaned data is JSON serializable:\n", json_output)
    except TypeError as e:
        print(f"\nError: Cleaned data is NOT JSON serializable: {e}")

    # --- Test create_session_summary_json ---
    print("\n--- Testing create_session_summary_json ---")
    mock_drift_report = {
        "schema_drift": {
            "added_columns": ["new_col"],
            "removed_columns": ["old_col"],
            "changed_columns": {"feature_x": {"old_type": "int64", "new_type": "float64"}}
        },
        "column_drift": {
            "feature_y": {
                "drift_score": np.float64(0.75), # Using np.float64 directly for mock
                "null_pct_old": np.float64(1.5),
                "null_pct_new": np.float64(10.0),
                "type_old": "int64",
                "type_new": "int64",
                "missing_values_drift": True,
                "data_type_changed": False,
                "mean_old": np.float64(100.0),
                "mean_new": np.float64(120.5)
            },
            "feature_z": {
                "drift_score": np.float64(0.20),
                "null_pct_old": np.float64(0.0),
                "null_pct_new": np.float64(0.0),
                "type_old": "object",
                "type_new": "object",
                "missing_values_drift": False,
                "data_type_changed": False,
                "category_drift": {
                    "new_categories": ["Gamma"],
                    "missing_categories": [],
                    "top_categories_old": {"Alpha": np.float64(0.6), "Beta": np.float64(0.4)},
                    "top_categories_new": {"Alpha": np.float64(0.5), "Beta": np.float64(0.3), "Gamma": np.float64(0.2)}
                }
            }
        }
    }

    mock_drift_summary_table = [
        {"Column": "feature_y", "Drift Score": 0.75, "Missing % (New)": 10.00, "Comments": "Mean: 100.00 -> 120.50"},
        {"Column": "feature_z", "Drift Score": 0.20, "Missing % (New)": 0.00, "Comments": "New categories: Gamma"}
    ]

    mock_ai_explanation = "The data shows significant drift in 'feature_y' (mean shift, increased missing values) and 'feature_z' (new categories). This could impact model performance and data integrity. Recommend investigating data sources and potentially retraining models."

    session_summary = create_session_summary_json(
        "baseline_data.csv",
        "current_data.csv",
        mock_drift_report,
        mock_drift_summary_table,
        mock_ai_explanation
    )

    print("\nGenerated Session Summary (first 500 chars):")
    print(json.dumps(session_summary, indent=2)[:500])

    try:
        json.dumps(session_summary)
        print("\nSession summary is successfully JSON serializable.")
    except TypeError as e:
        print(f"\nError: Session summary is NOT JSON serializable: {e}")

    print("\n--- Utils Example Complete ---")