
# 📊 AI-Powered Data Drift Monitor & Explainer

A **Streamlit web application** designed to help data engineers and analysts **detect, visualize, and understand data drift** in their datasets. It combines Python data profiling techniques with **Google Gemini AI** to provide human-readable insights into drift and its impact on business or ML pipelines.

---

## 🔍 Features

### 🧾 Dual Dataset Upload

* Upload your **baseline** (historical/stable) and **current** datasets (`.csv` format).
* Instantly preview datasets and view basic statistics.

### 📊 Automated Drift Detection

* **Per-Column Comparison**:

  * Stats: Mean, Std Dev, Null %, Unique values.
  * Categorical: Top categories, new/missing values.
* **Schema Drift**:

  * Automatically detects added, removed, or changed column types.
* **Drift Score**:

  * Quantitative measure per column to highlight significant changes.

### 🧠 Gemini-Powered Analysis (Quota-Safe)

* Sends **only concise summaries** to Gemini (optimized for free-tier).
* **Plain-English Explanation**:

  * What changed
  * Business/ML pipeline impact
  * Suggested remediation steps

### 📉 Visual Insights

* **Interactive Charts**:

  * Histograms (numerical) & bar charts (categorical)
* **Target Drift Visualization**:

  * Explore changes in your target variable (optional)

### 📤 Export Options

* **Download Full Session Summary**:

  * JSON report includes drift metrics, summaries, and AI explanation.

---

## 🛠 Tech Stack

| Component       | Tool / Library                            |
| --------------- | ----------------------------------------- |
| Frontend        | `Streamlit` (interactive web UI)          |
| Data Processing | `Pandas`, `NumPy`                         |
| AI Integration  | `Google Gemini API` (`gemini-1.0-pro`)    |
| Visualization   | `Matplotlib`, `Seaborn`, `Plotly Express` |
| Env Management  | `python-dotenv`                           |

---

## 🧪 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/data-drift-monitor.git
cd data-drift-monitor

# Create and activate virtual environment
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🔑 Setup Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app) and generate an API key.
2. Create a `.env` file in the root of the project.
3. Add your API key:

```env
GEMINI_API_KEY="your_google_gemini_api_key_here"
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

* Navigate to: [http://localhost:8501](http://localhost:8501)
* Upload `baseline_data.csv` and `current_data.csv` via sidebar
* Click **"📈 Analyze Data Drift"**
* Explore results:

  * **Summary & AI Insights**
  * **Visual Insights**
* Click **"✨ Get AI Explanation"** to activate Gemini
* Export session via **"📥 Download Session Summary (JSON)"**

---

## 📂 Project Structure

```
data-drift-monitor/
├── .env                    # Store Gemini API Key (DO NOT COMMIT)
├── app.py                  # Main Streamlit UI logic
├── drift_analysis.py       # Drift computation logic (PSI, schema, stats)
├── ai_logic.py             # Handles Gemini prompt + API calls
├── visualizer.py           # Interactive and static plots
├── utils.py                # JSON formatting, numpy conversion
├── requirements.txt        # Project dependencies
└── README.md               # You’re reading it!
```

---

## 🧠 Limitations

* Designed for **structured/tabular** data only.
* Optimized for **CSV** format (no Excel/Parquet support yet).
* Does **not** support time-series drift detection (e.g., trends/seasonality).
* Gemini responses are **condensed** to fit free-tier quota.

---

## ✅ Ideal For

* **Data Engineers**: Monitor pipeline health and schema changes
* **MLOps Teams**: Catch silent drift before it breaks your model
* **Data Analysts**: Compare “before” and “after” datasets for impact

---
