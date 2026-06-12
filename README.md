# 🏦 Bank Customer Churn Prediction & Risk Scoring

An interactive **Streamlit dashboard** for predicting customer churn risk in the banking sector. Features single and batch predictions, what-if analysis, model performance comparison, and session history tracking.

## 🚀 Live Demo (Streamlit Community Cloud)

**[Click here to launch the app](https://share.streamlit.io/)** — *Deploy after connecting to GitHub (see instructions below)*

---

## 📋 Project Structure

```
├── 01_churn_ml_pipeline.py      # ML training pipeline (run first)
├── 02_streamlit_dashboard.py    # Streamlit web app (run second)
├── European_Bank.csv            # Input dataset (10,001 records)
├── best_model.pkl               # Trained ML model
├── scaler.pkl                   # Feature scaler
├── feature_names.pkl            # Feature names list
├── model_metadata.json          # Model performance metrics
├── feature_importance.csv       # Feature importance scores
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── .streamlit/config.toml       # Streamlit theme & server config
├── quickStart.md                # Quick start guide (local)
└── README.md                    # This file
```

---

## 🛠️ Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (if artifacts are missing)
```bash
python 01_churn_ml_pipeline.py
```

### 3. Launch the Dashboard
```bash
streamlit run 02_streamlit_dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Deploy to Streamlit Community Cloud

### Prerequisites
- A [GitHub](https://github.com) account
- A [Streamlit Community Cloud](https://streamlit.io/cloud) account (free)

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - Bank Churn Prediction App"

# Create a repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[https://streamlit.io/cloud](https://streamlit.io/cloud)**
2. Click **"Get Started"** or **"New app"**
3. **Connect your GitHub account** (if not already connected)
4. **Select your repository** and branch (`main`)
5. **Set the main file path** to: `02_streamlit_dashboard.py`
6. Click **"Deploy"**

### Step 3: Done! 🎉

Your app will be live at:  
`https://YOUR_USERNAME-YOUR_REPO_NAME.streamlit.app`

Streamlit Cloud automatically:
- ✅ Installs dependencies from `requirements.txt`
- ✅ Loads all model artifacts from the repository
- ✅ Handles scaling, HTTPS, and updates

### Updating the App
After making changes, simply:
```bash
git add .
git commit -m "Your update message"
git push
```

Streamlit Cloud automatically rebuilds on every push.

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| **📊 Dashboard** | Model performance summary, key metrics, top churn drivers |
| **🎯 Risk Calculator** | Single customer prediction with what-if analysis |
| **📂 Batch Upload** | Upload CSV for bulk predictions + downloadable results |
| **📈 Model Performance** | Radar charts, bar comparisons, feature importance |
| **📜 History** | Session-based prediction tracking with trend charts |
| **❓ Help** | Documentation and metric explanations |

---

## ⚙️ Configuration

The `.streamlit/config.toml` file controls the app theme:
- **Primary Color:** `#1f77b4` (blue)
- **Background:** Light theme
- **Max Upload Size:** 50 MB

---

## 📄 License

This project was developed as part of an ML engineering internship.