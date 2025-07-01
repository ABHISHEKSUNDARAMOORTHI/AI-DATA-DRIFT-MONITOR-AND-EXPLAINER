import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import io # Import io for handling file uploads with pandas

# Import our custom modules with the correct function names
from drift_analysis import analyze_drift, generate_drift_summary_table, prepare_drift_summary_for_gemini
from visualizer import plot_drift_heatmap, plot_column_comparison, plot_target_drift
from ai_logic import get_gemini_analysis
from utils import create_session_summary_json # Import the utility for session saving

# Load environment variables (ensure your .env file is in the same directory as app.py)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Data Drift Monitor with Gemini AI",
    page_icon="📊",
    layout="wide", # Use wide layout for better visualization
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Theming and Better UI ---
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0F172A; /* Slate 950 */
        color: #F8FAFC; /* Slate 50 */
    }
    /* Sidebar styling */
    .st-emotion-cache-vk33gh, .st-emotion-cache-16txt3u { /* Target for sidebar background and text */
        background-color: #1E293B; /* Slate 800 */
        color: #F8FAFC; /* Slate 50 */
    }
    .st-emotion-cache-16txt3u { /* Specific target for sidebar text */
        color: #F8FAFC; /* Slate 50 */
    }
    /* Header/Title styling */
    h1 {
        color: #CBD5E1; /* Slate 300 */
        text-align: center;
    }
    h2, h3 {
        color: #E2E8F0; /* Slate 200 */
    }
    /* Button styling */
    .stButton > button {
        background-color: #4F46E5; /* Indigo 600 */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #6366F1; /* Indigo 500 */
    }
    /* File uploader styling */
    .st-emotion-cache-1f8jcil { /* File uploader container */
        background-color: #1E293B; /* Slate 800 */
        border: 2px dashed #475569; /* Slate 600 */
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .st-emotion-cache-1f8jcil p { /* File uploader text */
        color: #CBD5E1; /* Slate 300 */
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1E293B; /* Slate 800 */
        color: #F8FAFC; /* Slate 50 */
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #475569; /* Slate 600 */
    }
    .streamlit-expanderContent {
        background-color: #0F172A; /* Slate 950 */
        border: 1px solid #475569; /* Slate 600 */
        border-radius: 8px;
        padding: 15px;
        margin-top: -10px; /* Adjust to connect with header */
    }
    /* Info/Warning/Success boxes */
    div[data-testid="stInfo"] { /* Generic info box (used for DEBUG messages) */
        border-left: 5px solid #3B82F6; /* Blue 500 */
        background-color: #1E293B; /* Slate 800 */
        color: #93C5FD; /* Blue 300 */
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    div[data-testid="stWarning"] {
        border-left: 5px solid #EAB308; /* Amber 500 */
        background-color: #1E293B;
        color: #FDE68A; /* Amber 200 */
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    div[data-testid="stSuccess"] {
        border-left: 5px solid #22C55E; /* Green 500 */
        background-color: #1E293B;
        color: #86EFAD; /* Green 300 */
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    div[data-testid="stError"] {
        border-left: 5px solid #EF4444; /* Red 500 */
        background-color: #1E293B;
        color: #FCA5A5; /* Red 300 */
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    /* Dataframe styling */
    .st-emotion-cache-1kyxgrs { /* Dataframe container */
        border: 1px solid #475569;
        border-radius: 8px;
    }
    .dataframe {
        background-color: #1E293B; /* Slate 800 */
        color: #F8FAFC;
    }
    .dataframe th {
        background-color: #334155; /* Slate 700 */
        color: #CBD5E1;
    }
    .dataframe td {
        color: #F8FAFC;
    }
    /* Tabs styling */
    button[data-baseweb="tab"] {
        background-color: #1E293B; /* Slate 800 */
        color: #CBD5E1;
        border-bottom: 2px solid #475569; /* Slate 600 */
    }
    button[data-baseweb="tab"]:hover {
        background-color: #334155; /* Slate 700 */
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4F46E5; /* Indigo 600 */
        color: white;
        border-bottom: 2px solid #6366F1; /* Indigo 500 */
    }
    .ai-explanation-box { /* Apply markdown styling to AI output */
        background-color: #1E293B;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
# Initialize session state variables to store data and analysis results across reruns
if 'baseline_df' not in st.session_state:
    st.session_state.baseline_df = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'drift_report' not in st.session_state:
    st.session_state.drift_report = None
if 'drift_summary_table' not in st.session_state:
    st.session_state.drift_summary_table = None
if 'ai_explanation' not in st.session_state:
    st.session_state.ai_explanation = None
if 'baseline_filename' not in st.session_state:
    st.session_state.baseline_filename = "No file uploaded"
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = "No file uploaded"
# Add a session state variable for selected column for visualization
if 'selected_column_for_viz' not in st.session_state:
    st.session_state.selected_column_for_viz = "-- Select a column --"
if 'target_column_for_viz' not in st.session_state:
    st.session_state.target_column_for_viz = "-- Select a target column --"


# --- Title and Header ---
st.title("📊 AI-Powered Data Drift Monitor")
st.markdown("Monitor and explain data drift in your ML datasets with Google Gemini AI.")
st.markdown("---")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Datasets")

    st.info("Upload your **Baseline** (reference) and **Current** (new) datasets as CSV files.")

    # Helper function for loading CSV from bytes
    def load_csv_from_bytes(uploaded_file_bytes):
        return pd.read_csv(io.BytesIO(uploaded_file_bytes))

    baseline_file = st.file_uploader("Upload Baseline Dataset", type=["csv"], key="baseline_uploader")
    if baseline_file is not None:
        try:
            st.session_state.baseline_df = load_csv_from_bytes(baseline_file.getvalue())
            st.session_state.baseline_filename = baseline_file.name
            st.success(f"Baseline file '{st.session_state.baseline_filename}' loaded successfully!")
        except Exception as e:
            st.error(f"Error loading baseline file: {e}. Please ensure it's a valid CSV.")
            st.session_state.baseline_df = None
            st.session_state.baseline_filename = "Error loading file"

    current_file = st.file_uploader("Upload Current Dataset", type=["csv"], key="current_uploader")
    if current_file is not None:
        try:
            st.session_state.current_df = load_csv_from_bytes(current_file.getvalue())
            st.session_state.current_filename = current_file.name
            st.success(f"Current file '{st.session_state.current_filename}' loaded successfully!")
        except Exception as e:
            st.error(f"Error loading current file: {e}. Please ensure it's a valid CSV.")
            st.session_state.current_df = None
            st.session_state.current_filename = "Error loading file"

    st.markdown("---")
    st.subheader("Currently Loaded Files:")
    st.write(f"Baseline: `{st.session_state.baseline_filename}`")
    st.write(f"Current: `{st.session_state.current_filename}`")
    
    st.markdown("---")
    # Button to trigger analysis
    if st.session_state.baseline_df is not None and st.session_state.current_df is not None:
        if st.button("📈 Analyze Data Drift", help="Click to perform data drift analysis."):
            st.session_state.ai_explanation = None # Clear previous AI explanation on new analysis
            st.session_state.selected_column_for_viz = "-- Select a column --" # Reset dropdown
            st.session_state.target_column_for_viz = "-- Select a target column --" # Reset target dropdown
            
            with st.spinner("Analyzing data drift... This may take a moment."):
                try:
                    st.session_state.drift_report = analyze_drift(st.session_state.baseline_df, st.session_state.current_df)
                    st.session_state.drift_summary_table = generate_drift_summary_table(st.session_state.drift_report)
                    st.success("Data Drift Analysis Complete! Check the tabs for results.")
                except Exception as e:
                    st.error(f"An error occurred during drift analysis: {e}. Please check your dataset format and content.")
                    st.session_state.drift_report = None
                    st.session_state.drift_summary_table = None
                
    else:
        st.warning("Please upload both Baseline and Current datasets in CSV format to enable analysis.")

    # --- Debugging & Advanced Options (Expanded by default) ---
    with st.expander("⚙️ Debug & Advanced Options"):
        st.subheader("Debug Information")
        if GEMINI_API_KEY:
            st.success("Gemini API Key loaded from environment variable.")
        else:
            st.warning("Gemini API Key not found. Please ensure it's set in your `.env` file.")
            st.info("You can get a Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).")

        st.markdown("---")
        st.subheader("Session Summary Export")
        st.info("Exports current analysis results to a JSON file.")
        
        # Add a placeholder for download button
        download_button_placeholder = st.empty()

        if st.session_state.drift_report:
            # Prepare summary for export
            session_summary_data = create_session_summary_json(
                baseline_filename=st.session_state.baseline_filename,
                current_filename=st.session_state.current_filename,
                drift_report=st.session_state.drift_report,
                drift_summary_table=st.session_state.drift_summary_table.to_dict('records') if st.session_state.drift_summary_table is not None else None,
                ai_explanation=st.session_state.ai_explanation
            )
            
            # Generate a dynamic filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_report_{timestamp}.json"
            
            # Convert dictionary to JSON string
            json_string = json.dumps(session_summary_data, indent=4)
            
            # Display download button
            download_button_placeholder.download_button(
                label="📥 Download Session Summary (JSON)",
                data=json_string,
                file_name=filename,
                mime="application/json",
                help="Download a JSON file containing the drift report, summary table, and AI explanation."
            )
        else:
            download_button_placeholder.info("Perform analysis first to enable session summary download.")


# --- Main Content Area ---
if st.session_state.drift_report is None:
    st.info("Upload your datasets in the sidebar and click 'Analyze Data Drift' to begin the analysis.")
    
    # Display preview of uploaded data if available
    col1_p, col2_p = st.columns(2)
    with col1_p:
        if st.session_state.baseline_df is not None:
            st.subheader("Baseline Data Preview:")
            st.dataframe(st.session_state.baseline_df.head(), use_container_width=True)
            st.write(f"Shape: {st.session_state.baseline_df.shape}")
    with col2_p:
        if st.session_state.current_df is not None:
            st.subheader("Current Data Preview:")
            st.dataframe(st.session_state.current_df.head(), use_container_width=True)
            st.write(f"Shape: {st.session_state.current_df.shape}")

else:
    st.header("Drift Analysis Results")

    # Create tabs for different views of the results
    tab1, tab2 = st.tabs(["Summary & AI Insights", "Visual Insights"])

    with tab1:
        st.subheader("Drift Summary Table")
        st.dataframe(st.session_state.drift_summary_table, use_container_width=True)

        st.markdown("---")
        st.subheader("Schema Drift Detected")
        # Ensure schema_drift key exists and is a dictionary, default to empty dict if not
        schema_drift = st.session_state.drift_report.get('schema_drift', {}) 
        
        if schema_drift.get('added_columns') or \
           schema_drift.get('removed_columns') or \
           schema_drift.get('changed_columns'):
            
            if schema_drift.get('added_columns'):
                st.error(f"**Added Columns:** {', '.join(schema_drift['added_columns'])}")
            if schema_drift.get('removed_columns'):
                st.error(f"**Removed Columns:** {', '.join(schema_drift['removed_columns'])}")
            if schema_drift.get('changed_columns'):
                st.warning("**Changed Column Types:**")
                for col, types in schema_drift['changed_columns'].items():
                    st.info(f" - `{col}`: Changed from `{types['old_type']}` to `{types['new_type']}`")
        else:
            st.success("No significant schema drift detected.")

        st.markdown("---")
        st.subheader("AI-Powered Explanation")
        if st.session_state.ai_explanation is None:
            if st.button("✨ Get AI Explanation", help="Ask Gemini AI to summarize and explain the drift."):
                with st.spinner("Generating AI explanation..."):
                    try:
                        # Prepare summary for AI (ensures JSON serializable and concise)
                        summary_for_gemini = prepare_drift_summary_for_gemini(st.session_state.drift_report)
                        st.session_state.ai_explanation = get_gemini_analysis(summary_for_gemini, GEMINI_API_KEY)
                    except Exception as e:
                        st.session_state.ai_explanation = f"An error occurred while getting AI explanation: {e}"
            else:
                st.info("Click 'Get AI Explanation' to receive an AI-powered summary of the data drift.")
        
        if st.session_state.ai_explanation:
            st.markdown(
                f'<div class="ai-explanation-box">{st.session_state.ai_explanation}</div>', 
                unsafe_allow_html=True
            )

    with tab2:
        st.subheader("Visual Insights")

        # Select a column for detailed visualization
        all_columns = sorted(list(set(st.session_state.baseline_df.columns).union(st.session_state.current_df.columns)))
        
        # Use a consistent key for the selectbox to persist selection across reruns
        st.session_state.selected_column_for_viz = st.selectbox(
            "Select a column for detailed distribution comparison:",
            ["-- Select a column --"] + all_columns,
            key="column_select_box", # Key for the selectbox
            index=(all_columns.index(st.session_state.selected_column_for_viz) + 1 if st.session_state.selected_column_for_viz in all_columns else 0)
        )

        if st.session_state.selected_column_for_viz != "-- Select a column --":
            selected_col = st.session_state.selected_column_for_viz
            
            # Check if column exists in both DFs, warn if not
            col_in_baseline = selected_col in st.session_state.baseline_df.columns
            col_in_current = selected_col in st.session_state.current_df.columns

            if not col_in_baseline:
                st.warning(f"Column '{selected_col}' is *missing* from the Baseline dataset.")
            if not col_in_current:
                st.warning(f"Column '{selected_col}' is *missing* from the Current dataset.")

            if col_in_baseline and col_in_current:
                # Plot detailed column comparison
                try:
                    fig_col_comparison = plot_column_comparison(
                        st.session_state.baseline_df, 
                        st.session_state.current_df, 
                        selected_col
                    )
                    if fig_col_comparison:
                        st.plotly_chart(fig_col_comparison, use_container_width=True)
                    else:
                        st.info(f"Could not generate comparison plot for '{selected_col}'.")
                except Exception as e:
                    st.error(f"Error generating plot for '{selected_col}': {e}")
            elif not col_in_baseline and not col_in_current:
                st.warning(f"Column '{selected_col}' is missing from *both* datasets. Cannot compare.")
            else:
                st.info(f"Cannot perform a full comparison as '{selected_col}' is only present in one dataset. You selected it, but it's not present in both for a direct comparison.")
                if col_in_baseline:
                    st.dataframe(st.session_state.baseline_df[[selected_col]].describe(), use_container_width=True)
                if col_in_current:
                    st.dataframe(st.session_state.current_df[[selected_col]].describe(), use_container_width=True)


        st.markdown("---")
        st.subheader("Overview: Heatmap of Drift Scores")
        # Plot overall drift heatmap
        if st.session_state.drift_report.get('column_drift'):
            try:
                heatmap_fig = plot_drift_heatmap(st.session_state.drift_report)
                if heatmap_fig:
                    st.pyplot(heatmap_fig) # Streamlit uses st.pyplot for matplotlib figures
                    st.info("This heatmap visualizes the drift score for each common column. Higher scores indicate more significant drift.")
                else:
                    st.warning("Could not generate drift heatmap.")
            except Exception as e:
                st.error(f"Error generating drift heatmap: {e}")
        else:
            st.info("No column drift data available to generate a heatmap.")

        st.markdown("---")
        st.subheader("Target Variable Drift Analysis")
        # Select a target column for visualization
        # Columns present in both dataframes
        common_cols_for_target = list(set(st.session_state.baseline_df.columns) & set(st.session_state.current_df.columns))

        st.session_state.target_column_for_viz = st.selectbox(
            "Select your target variable (must be in both datasets):",
            ["-- Select a target column --"] + sorted(common_cols_for_target),
            key="target_column_select_box",
            index=(sorted(common_cols_for_target).index(st.session_state.target_column_for_viz) + 1 if st.session_state.target_column_for_viz in common_cols_for_target else 0)
        )

        if st.session_state.target_column_for_viz != "-- Select a target column --":
            target_col = st.session_state.target_column_for_viz
            try:
                target_drift_fig = plot_target_drift(
                    st.session_state.baseline_df,
                    st.session_state.current_df,
                    target_col
                )
                if target_drift_fig:
                    st.pyplot(target_drift_fig)
                    st.info(f"Distribution of the target variable '{target_col}' in baseline versus current datasets.")
                else:
                    st.warning(f"Could not generate target drift plot for '{target_col}'.")
            except Exception as e:
                st.error(f"Error generating target drift plot for '{target_col}': {e}")
        else:
            st.info("Select a target column to visualize its distribution drift.")