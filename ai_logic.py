import google.generativeai as genai
import os
import json
import logging
# Import GoogleAPIError for more robust error handling
from google.api_core.exceptions import GoogleAPIError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_gemini(api_key):
    """
    Configures the Google Gemini API with the provided key.
    Returns True on success, False on failure.
    """
    if not api_key or api_key == "your_google_gemini_api_key_here":
        logging.error("Gemini API Key is missing or is a placeholder. Please set GEMINI_API_KEY in your .env file.")
        return False
    
    # Check if the API key looks like a valid key (basic format check)
    if not api_key.startswith("AIza"):
        logging.error("Gemini API Key format appears invalid. It should start with 'AIza'.")
        return False

    try:
        genai.configure(api_key=api_key)
        logging.info("Gemini API configured successfully.")
        # Optional: Test a very basic call to ensure connectivity
        try:
            # list_models() is a good way to check connectivity without generating content
            list(genai.list_models()) 
            logging.info("Successfully connected to Gemini API (models listed).")
            return True
        except Exception as e: # Catch any exception during model listing
            logging.error(f"Failed to connect/list models with provided API key, even after configure: {e}")
            return False
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {e}")
        return False

def get_gemini_analysis(drift_summary_for_ai, api_key):
    """
    Sends the concise drift summary to Google Gemini and gets an explanation.

    Args:
        drift_summary_for_ai (dict): A concise dictionary summary of data drift,
                                     optimized for Gemini's input. This dict
                                     should already have NumPy types converted
                                     to standard Python types.
        api_key (str): Your Google Gemini API key.

    Returns:
        str: Gemini's explanation in Markdown format, or a detailed error message.
    """
    # 1. API Key Check and Configuration
    if not configure_gemini(api_key):
        return "🚫 **AI Analysis Disabled:** Invalid or missing Gemini API Key. Please set `GEMINI_API_KEY` correctly in your `.env` file and ensure it's valid."

    try:
        # Initialize Gemini model - try multiple model names in order of preference
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.0-pro-latest']
        model = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                logging.info(f"Successfully initialized model: {model_name}")
                break
            except Exception as e:
                logging.warning(f"Failed to initialize model {model_name}: {e}")
                continue
        
        if model is None:
            logging.error("Failed to initialize any Gemini model")
            return "❌ **AI Analysis Error:** Could not initialize any available Gemini model. Please check available models using the test script."

        # Prompt preparation
        try:
            summary_json_str = json.dumps(drift_summary_for_ai, indent=2)
        except TypeError as e:
            logging.error(f"TypeError: drift_summary_for_ai is not JSON serializable: {e}")
            return f"❌ **AI Analysis Error:** Internal data issue. Drift summary is not JSON serializable. Error: `{e}`"
        except Exception as e:
            logging.error(f"Unexpected error during JSON serialization of drift_summary: {e}")
            return f"❌ **AI Analysis Error:** Unexpected issue preparing data for AI. Error: `{e}`"

        prompt_template = f"""
Analyze the following data drift summary between a baseline dataset and a new dataset.
Provide a clear, concise explanation in Markdown format, covering:

1.  **What changed:** Summarize the key data changes (e.g., shifts in numerical means, categorical distributions, null percentages, schema changes). Focus on columns with significant drift scores (score >= 0.4) and notable schema/missing value changes.
2.  **Potential Impact:** Explain the likely business or ML pipeline impacts of these changes.
3.  **Suggested Remediation:** Propose actionable steps to address the identified drift.

Drift Summary (JSON):
```json
{summary_json_str}
```

Ensure your response is directly in Markdown, suitable for display, and avoid conversational filler like "Hello!", "Here is the analysis", etc. Start directly with the explanation.
"""

        # Log prompt length and content for debugging
        logging.info(f"Prompt length: {len(prompt_template)} characters.")
        if len(prompt_template) > 5000: # Example threshold, Gemini has token limits
            logging.warning(f"Prompt is very long ({len(prompt_template)} chars). This might lead to issues or high token usage.")
            # Consider truncating the summary_json_str if it's excessively large

        logging.info(f"Sending prompt to Gemini (first 500 chars):\n{prompt_template[:500]}...")

        # Generate content with generation configuration for better control
        response = model.generate_content(
            prompt_template,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,  # Lower temperature for less randomness, more factual output
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024 # Limit token usage for free tier efficiency
            )
        )
        
        # 2. Check for successful candidates in the response
        if response.candidates:
            # Ensure the structure is as expected: candidates[0].content.parts[0].text
            if hasattr(response.candidates[0], 'content') and \
               hasattr(response.candidates[0].content, 'parts') and \
               response.candidates[0].content.parts:
                gemini_text = response.candidates[0].content.parts[0].text
                logging.info("Successfully received AI explanation from Gemini.")
                return gemini_text.strip()
            else:
                logging.error(f"Gemini response has candidates but unexpected structure. Response: {response}")
                return "❌ **AI Analysis Failed:** Gemini returned an unexpected response structure. Please try again."
        else:
            # No candidates usually means a safety block or other internal model issue
            feedback = response.prompt_feedback
            safety_ratings_str = ""
            if feedback and feedback.safety_ratings:
                safety_ratings_str = ", ".join([f"{s.category.name}: {s.probability.name}" for s in feedback.safety_ratings])
            
            logging.warning(f"Gemini response had no candidates. Prompt feedback: {feedback}")
            return f"❌ **AI Analysis Failed:** Gemini did not return a valid response. This might be due to safety filters ({safety_ratings_str}), empty input, or other API issues. Please review your input data for sensitive content or try again. (No Candidates / Feedback: {feedback})"

    # 3. Handle specific Gemini API exceptions
    except genai.types.BlockedPromptException as e:
        logging.error(f"Gemini API BlockedPromptException: {e}. Prompt feedback: {e.response.prompt_feedback if hasattr(e.response, 'prompt_feedback') else 'N/A'}")
        safety_ratings_str = ""
        if hasattr(e.response, 'prompt_feedback') and e.response.prompt_feedback.safety_ratings:
             safety_ratings_str = ", ".join([f"{s.category.name}: {s.probability.name}" for s in e.response.prompt_feedback.safety_ratings])
        return f"❌ **AI Analysis Blocked:** Your prompt was blocked due to safety concerns or policy violations. Please review the input data for sensitive content. (Details: {safety_ratings_str})"
    except genai.types.StopCandidateException as e:
        logging.error(f"Gemini API StopCandidateException: {e}")
        return "❌ **AI Analysis Halted:** Gemini stopped generating a response prematurely. This can happen if max output tokens are reached or due to internal model reasons. (Try reducing prompt size or increasing max_output_tokens slightly)"
    except GoogleAPIError as e: # Catch general Google API errors
        logging.error(f"GoogleAPIError occurred during Gemini API call: {e}", exc_info=True)
        return f"❌ **AI Analysis API Error:** Failed to connect to Gemini API or an internal API error occurred. Error: `{e}`. Check your internet connection or try again later."
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred during Gemini API call: {e}", exc_info=True)
        return f"❌ **AI Analysis Error:** An unexpected error occurred: `{e}`. Please check your internet connection, API key, or try again with different input data."


# Example Usage (for testing this file directly)

if __name__ == '__main__':
    print("Running ai_logic.py example...")

    # Load API key from .env (for standalone testing)
    from dotenv import load_dotenv
    load_dotenv()
    test_api_key = os.getenv("GEMINI_API_KEY")

    if not test_api_key or test_api_key == "your_google_gemini_api_key_here":
        print("WARNING: GEMINI_API_KEY is not set. Cannot run full AI logic test.")
        print("Please set your API key in the .env file to test this module.")
    else:
        # --- Test API Key Configuration ---
        print("\n--- Testing Gemini API Configuration ---")
        if configure_gemini(test_api_key):
            print("Gemini API configured and tested successfully.")
        else:
            print("Gemini API configuration failed. Check logs above.")
            # exit() # Exit if configuration fails for direct test

        # --- Optional: List available models (useful for debugging 404 errors) ---
        try:
            print("\n--- Listing Available Gemini Models ---")
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
                    print(f"Model: {m.name}, Description: {m.description}")
            print("--------------------------------------")
            if available_models:
                print(f"Available models for generateContent: {', '.join(available_models)}")
            else:
                print("No models found that support generateContent")
        except Exception as e:
            print(f"Error listing models: {e}. This might indicate an API key issue or network problem even after successful configure.")
        # --- End of optional model listing ---

        # Simulate a concise drift summary from drift_analysis.py
        # This example is simplified but represents the structure 'prepare_drift_summary_for_gemini' would output
        # It assumes numpy types are already converted to standard Python types.
        sample_drift_summary = {
            "schema_drift_summary": {
                "added_columns": ["new_feature"],
                "removed_columns": [],
                "changed_columns": {
                    "customer_age": {
                        "old_type": "int64",
                        "new_type": "float64"
                    }
                }
            },
            "column_drift_summary": {
                "customer_age": {
                    "drift_score": 0.88,
                    "null_pct_old": 2.1,
                    "null_pct_new": 12.7,
                    "mean_old": 35.2,
                    "mean_new": 42.8,
                    "data_type_old": "int64",
                    "data_type_new": "float64",
                    "missing_values_drift_detected": True,
                    "statistics_drift": {
                        "mean_old": 35.2,
                        "mean_new": 42.8
                    }
                },
                "gender": {
                    "drift_score": 0.65,
                    "null_pct_old": 0.0,
                    "null_pct_new": 0.0,
                    "data_type_old": "object",
                    "data_type_new": "object",
                    "missing_values_drift_detected": False,
                    "statistics_drift": {
                        "top_categories_old": {"Male": 53.0, "Female": 47.0},
                        "top_categories_new": {"Female": 67.0, "Male": 33.0},
                        "new_categories": ["Non-binary"],
                        "missing_categories": []
                    }
                },
                "product_id": { # Example of a column with low drift but present
                    "drift_score": 0.15,
                    "null_pct_old": 0.0,
                    "null_pct_new": 0.0,
                    "data_type_old": "int64",
                    "data_type_new": "int64",
                    "missing_values_drift_detected": False,
                    "statistics_drift": {
                        "mean_old": 100.0,
                        "mean_new": 100.5
                    }
                }
            }
        }

        print("\n--- Requesting AI Analysis ---")
        ai_explanation_text = get_gemini_analysis(sample_drift_summary, test_api_key)
        print("\n--- AI Explanation ---")
        print(ai_explanation_text)

    print("\nAI Logic Example Complete.")