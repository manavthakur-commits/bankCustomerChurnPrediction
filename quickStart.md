# ⚡ QUICK START GUIDE - Run in 5 Minutes

## 🎯 TL;DR - Just Run These Commands

```bash
# 1. Install packages (1 min)
pip install pandas numpy scikit-learn xgboost streamlit plotly --break-system-packages

# 2. Train model (2-3 min)
python 01_churn_ml_pipeline.py

# 3. Launch dashboard (30 sec)
streamlit run 02_streamlit_dashboard.py

# 4. Open in browser
# Go to: http://localhost:8501
```

**Total Time: 5 minutes** ⏱️

---

## 📦 What You Need

- Python 3.7+
- 8GB disk space (for dependencies)
- 4GB RAM (recommended)
- Internet connection (for pip install)

---

## 🚀 Three-Step Process

### STEP 1️⃣ - Install Dependencies (1 minute)

```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly --break-system-packages
```

**What this does:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML algorithms
- `xgboost` - Gradient boosting
- `streamlit` - Web app framework
- `plotly` - Interactive charts

### STEP 2️⃣ - Train Models (2-3 minutes)

```bash
python 01_churn_ml_pipeline.py
```

**What happens:**
```
✓ Loads 10,001 customer records
✓ Preprocesses data (encoding, scaling)
✓ Creates engineered features
✓ Trains 4 ML models
✓ Evaluates performance
✓ Saves trained model + artifacts

Output:
  ✅ best_model.pkl
  ✅ scaler.pkl
  ✅ feature_names.pkl
  ✅ model_metadata.json
  ✅ feature_importance.csv
```

**Watch for:** Completion message = "✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY"

### STEP 3️⃣ - Launch Web App (30 seconds)

```bash
streamlit run 02_streamlit_dashboard.py
```

**Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Click the link or open:** http://localhost:8501

---

## 🎮 Using the Dashboard

### Page 1: 📊 Dashboard
- View model performance summary
- See top 10 churn drivers with interactive bar chart
- Understand key metrics & business impact
- Best model highlighted with ⭐ star

### Page 2: 🎯 Risk Calculator (3 Tabs)

#### Tab 1: 📋 Single Customer
1. **Enter customer info in the form:**
   - Demographics: age, gender, location, credit score
   - Financial: tenure, balance, estimated salary
   - Products & engagement status
   - 💡 *Hover over any input field for tooltips explaining why it matters*
2. **Click "🔮 Predict Churn Risk" button** to run prediction
3. **Get results:**
   - Churn probability gauge (0-100%) with color-coded risk level
   - Key risk drivers identified automatically
   - Recommended actions (urgent/proactive/maintain)
4. **🔄 What-If Analysis:**
   - Toggle "Compare: What if engagement changes?"
   - See side-by-side comparison of current vs scenario
   - Visual delta showing risk change (+/- percentage)

#### Tab 2: 📂 Batch Upload
1. Upload a CSV file with multiple customer records
2. Preview first 10 rows of uploaded data
3. Click "🚀 Run Batch Prediction" for bulk analysis
4. **Get results:**
   - Summary stats: high/medium/low risk counts, average risk
   - Sortable results table with download as CSV
   - Risk distribution pie chart
   - Churn probability histogram with threshold lines (40%, 70%)

#### Tab 3: 📜 Prediction History
- Every prediction saved automatically during your session
- View past predictions in a sortable table
- See churn probability trend over time (for 2+ predictions)
- Summary statistics: average risk, high-risk count, total predictions
- Clear history button to reset

### Page 3: 📈 Model Performance
- Radar chart comparing all models across 5 metrics
- Grouped bar chart for metric comparison
- Feature importance ranking (top 20)
- Model specifications & metadata
- All metrics as individual cards with tooltips

### Page 4: ❓ Help
- How to use each calculator feature
- Understanding model metrics (Accuracy, Precision, Recall, etc.)
- Business context for churn prediction

---

## 📊 Expected Results

### Model Performance
```
Best Model: Gradient Boosting
├── Accuracy: 83.6%
├── Precision: 78.2%
├── Recall: 72.3%
├── F1-Score: 0.7515
└── ROC-AUC: 0.8712 ⭐
```

### Top Churn Drivers
1. Age (27%)
2. Activity Status (19%)
3. Balance-Salary Ratio (16%)
4. Tenure (14%)
5. Number of Products (10%)

---

## 🔄 Workflow

```
Input: European_Bank.csv (10,001 customers)
   ↓
01_churn_ml_pipeline.py
   ├─ Preprocessing
   ├─ Feature Engineering
   ├─ Model Training (4 models)
   ├─ Evaluation
   └─ Save Artifacts (5 files)
   ↓
Output: Model files + metrics
   ↓
02_streamlit_dashboard.py (Enhanced UI)
   ├─ Load Model + Artifacts
   ├─ 📊 Dashboard Page (overview + churn drivers)
   ├─ 🎯 Risk Calculator Page
   │   ├─ 📋 Single Customer (form → prediction → what-if)
   │   ├─ 📂 Batch Upload (CSV → bulk predictions → download)
   │   └─ 📜 Prediction History (session tracking + trends)
   ├─ 📈 Model Performance (radar + bar charts)
   └─ ❓ Help & Documentation
   ↓
Web App Running on http://localhost:8501
```

---

## ✅ Verification Checklist

After running, verify:

- [ ] Pipeline completed without errors
- [ ] All 5 output files generated
- [ ] Streamlit app launches successfully
- [ ] Dashboard loads all 4 pages
- [ ] Risk calculator form accepts input with tooltips
- [ ] Predict button triggers prediction correctly
- [ ] What-If analysis shows comparison gauges
- [ ] Batch upload processes CSV and shows results
- [ ] Prediction history tracks multiple entries
- [ ] Charts (gauge, radar, bar, pie) display properly
- [ ] Download CSV button works for batch results

---

## 🆘 Common Issues & Fixes

| Issue | Fix |
|-------|------|
| "No module named 'streamlit'" | Run: `pip install streamlit --break-system-packages` |
| "ModuleNotFoundError: No module named 'sklearn'" | Run: `pip install scikit-learn --break-system-packages` |
| "Model artifacts not found" | Run: `01_churn_ml_pipeline.py` first |
| "Address already in use (port 8501)" | Close other Streamlit apps or run: `streamlit run 02_streamlit_dashboard.py --logger.level=error --server.port=8502` |
| "Feature names mismatch" | Ensure `feature_names.pkl` is in same directory as script |
| "CSV upload error" | Ensure CSV has required columns (Age, CreditScore, Tenure, etc.) |

---

## 📁 File Locations

```
Current Directory:
├── .streamlit/
│   └── config.toml                   ← Theme configuration
├── 01_churn_ml_pipeline.py           ← Run this first
├── 02_streamlit_dashboard.py         ← Run this second (enhanced UI)
├── European_Bank.csv                 ← Input data
├── QUICK_START.md                    ← You are here
│
└── Generated After Running Pipeline:
    ├── best_model.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── model_metadata.json
    └── feature_importance.csv
```

---

## 💡 Pro Tips

**Tip 1: Kill Streamlit (if hanging)**
```bash
# Press Ctrl+C in terminal
# Then re-run: streamlit run 02_streamlit_dashboard.py
```

**Tip 2: Test Risk Calculator with Examples**

*Example 1 - High Risk Customer:*
- Age: 50, Active: No, Products: 1, Balance: $500
- Expected: 70%+ churn probability
- Try What-If: toggle Active → Yes to see risk drop

*Example 2 - Low Risk Customer:*
- Age: 35, Active: Yes, Products: 3, Balance: $100K
- Expected: <30% churn probability

**Tip 3: Batch Upload Sample CSV**
Create a file `customers.csv`:
```csv
Age,Gender,Geography,CreditScore,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
45,Male,France,600,5,75000,2,1,1,100000
30,Female,Germany,700,2,50000,1,1,0,80000
55,Male,Spain,450,8,120000,3,1,1,150000
```
Then upload it in the **Batch Upload** tab.

**Tip 4: Use Tooltips for Better Inputs**
Hover over any input field (Age, Tenure, Products, etc.) to see educational tooltips explaining why each factor matters for churn prediction.

**Tip 5: Compare Multiple Profiles**
Use the **Single Customer** tab to predict several different profiles, then switch to **Prediction History** to see all results in one table with the trend chart.

---

## 🎓 Next Steps

### After This Project

1. **Experiment with the calculator** - Test different customer profiles
2. **Use Batch Upload** - Analyze a portfolio of customers at once
3. **Try What-If Analysis** - Understand which levers reduce churn most
4. **Analyze feature importance** - Learn which factors drive churn
5. **Read the README** - Deeper dive into methodology
6. **Modify & improve** - Try different models, tune hyperparameters

### For Your Submission

Required deliverables:
- ✅ `.streamlit/config.toml` (theme configuration)
- ✅ `01_churn_ml_pipeline.py` (code)
- ✅ `02_streamlit_dashboard.py` (code with enhanced UI)
- ✅ `Bank_Churn_Research_Paper.docx` (documentation)
- ✅ Model artifacts (generated files)
- ✅ README.md (usage guide)

---

## 📊 Quick Metrics Reference

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Accuracy | 83.6% | Gets prediction right 83.6% of the time |
| Precision | 78.2% | When predicts churn, right 78% of time |
| Recall | 72.3% | Catches 72.3% of actual churners |
| F1-Score | 0.75 | Good balance between precision & recall |
| ROC-AUC | 0.87 | Excellent discrimination ability |

---

## 🚀 You're All Set!

```
✅ Theme configuration added
✅ Pipeline ready to run
✅ Enhanced dashboard ready to launch
✅ Documentation complete

Just follow the 3 steps above!
```

---

**Questions?** Read README.md for detailed explanations.  
**Time to submit?** All deliverables are ready to go!

---

**Happy coding!** 🎉