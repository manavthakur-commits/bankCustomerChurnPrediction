"""
Bank Customer Churn Prediction - Streamlit Dashboard (Enhanced UI)
Author: ML Intern
Date: 2025
Description: Interactive web application for churn prediction and risk scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bank Churn Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
        font-size: 1.2em;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
        font-size: 1.2em;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.2em;
    }
    .history-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 5px 0;
    }
    .comparison-delta-positive {
        color: #d62728;
        font-weight: bold;
    }
    .comparison-delta-negative {
        color: #2ca02c;
        font-weight: bold;
    }
    .prediction-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .batch-results {
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model and preprocessing objects"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, feature_names, metadata
    except FileNotFoundError:
        st.error("❌ Model artifacts not found. Please run the ML pipeline first.")
        st.stop()

@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    try:
        return pd.read_csv('feature_importance.csv')
    except FileNotFoundError:
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_category(probability):
    """Categorize churn risk based on probability"""
    if probability >= 0.7:
        return "🔴 HIGH RISK", "risk-high"
    elif probability >= 0.4:
        return "🟠 MEDIUM RISK", "risk-medium"
    else:
        return "🟢 LOW RISK", "risk-low"

def get_risk_color(probability):
    """Return hex color for risk level"""
    if probability >= 0.7:
        return "#d62728"
    elif probability >= 0.4:
        return "#ff7f0e"
    else:
        return "#2ca02c"

def prepare_input_features(input_dict, scaler, feature_names):
    """Convert user input to feature vector"""
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all features are present and preserve training order
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    
    # Align input to the scaler's fitted feature order before scaling
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
        input_df = input_df.reindex(columns=scaler_features, fill_value=0)
        input_df[scaler_features] = scaler.transform(input_df[scaler_features])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
    else:
        numerical_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df[feature_names]

def predict_churn(input_features, model):
    """Make churn prediction"""
    probability = model.predict_proba(input_features)[0, 1]
    prediction = model.predict(input_features)[0]
    return probability, prediction

def build_input_dict(age, gender, geography, credit_score, tenure, balance,
                     estimated_salary, num_products, has_credit_card, is_active):
    """Build feature dictionary from user inputs"""
    return {
        'Age': age,
        'CreditScore': credit_score,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active),
        'EstimatedSalary': estimated_salary,
        'Geography_Germany': 1 if geography == 'Germany' else 0,
        'Geography_Spain': 1 if geography == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Balance_Salary_Ratio': balance / (estimated_salary + 1),
        'Product_Density': num_products / (tenure + 1),
        'Engagement_Product_Score': int(is_active) * num_products,
        'Age_Tenure_Interaction': age * tenure,
        'Age_Group_Young': 1 if age < 30 else 0,
        'Age_Group_Senior': 1 if age >= 55 else 0,
        'CreditScore_Normalized': (credit_score - 350) / (850 - 350)
    }

def process_single_prediction(age, gender, geography, credit_score, tenure, balance,
                              estimated_salary, num_products, has_credit_card, is_active,
                              model, scaler, feature_names):
    """Process a single prediction and return results"""
    input_data = build_input_dict(
        age, gender, geography, credit_score, tenure, balance,
        estimated_salary, num_products, has_credit_card, is_active
    )
    input_features = prepare_input_features(input_data, scaler, feature_names)
    churn_probability, churn_prediction = predict_churn(input_features, model)
    risk_label, risk_class = get_risk_category(churn_probability)
    return input_data, churn_probability, churn_prediction, risk_label, risk_class

def create_gauge_chart(probability, height=350):
    """Create a gauge chart for churn risk"""
    fig_gauge = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk %"},
        delta={'reference': 50, 'increasing': {'color': "#d62728"}, 'decreasing': {'color': "#2ca02c"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': get_risk_color(probability)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 40], 'color': "#e8f5e9"},
                {'range': [40, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    )])
    fig_gauge.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig_gauge

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'use_what_if' not in st.session_state:
        st.session_state.use_what_if = False
    if 'baseline_prediction' not in st.session_state:
        st.session_state.baseline_prediction = None
    if 'last_inputs' not in st.session_state:
        st.session_state.last_inputs = None

# ============================================================================
# RENDER FUNCTIONS FOR EACH PAGE
# ============================================================================

def render_dashboard_page(metadata, feature_importance_df):
    """Page 1: Dashboard overview"""
    st.title("🏦 Bank Customer Churn Risk Scoring System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Status", "✅ Active", "Deployed",
                  help="Current operational status of the churn prediction model")
    with col2:
        st.metric("Best Model", metadata['best_model'], "F1-Optimized",
                  help=f"Best performing model selected from {len(metadata['model_metrics'])} candidates")
    with col3:
        st.metric("Last Updated", metadata['timestamp'][:10], "Ready",
                  help="Date when the model was last retrained")
    
    st.markdown("---")
    st.subheader("📌 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.info(
            """
            **What This Model Does:**
            • Predicts customer churn probability (0-1 scale)
            • Identifies high-risk customers for proactive retention
            • Explains key drivers of churn behavior
            • Enables scenario-based what-if analysis
            """
        )
    
    with insights_col2:
        st.warning(
            """
            **Business Impact:**
            • Early identification of at-risk customers
            • Targeted retention campaigns reduce churn cost
            • Personalized offers based on risk profiles
            • Improved customer lifetime value (CLV)
            """
        )
    
    st.markdown("---")
    st.subheader("🎯 Model Performance Summary")
    
    # Model comparison table with color coding
    metrics_data = []
    best_f1 = max(m['f1_score'] for m in metadata['model_metrics'].values())
    
    for model_name, metrics in metadata['model_metrics'].items():
        is_best = metrics['f1_score'] == best_f1
        metrics_data.append({
            'Model': f"⭐ {model_name}" if is_best else model_name,
            'Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🔝 Top 10 Churn Drivers")
    
    if feature_importance_df is not None:
        top_features = feature_importance_df.head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    line=dict(color='rgba(0,0,0,0.3)', width=1)
                ),
                text=top_features['importance'].round(3),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Feature Importance - Top 10 Churn Drivers",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=450,
            showlegend=False,
            margin=dict(l=10, r=40, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_risk_calculator_page(model, scaler, feature_names):
    """Page 2: Risk Calculator with form, history, what-if, and batch upload"""
    st.title("🎯 Customer Churn Risk Calculator")
    st.markdown("---")
    
    tab_calc, tab_batch, tab_history = st.tabs([
        "📋 Single Customer", "📂 Batch Upload", "📜 Prediction History"
    ])
    
    # ====================================================================
    # TAB 1: SINGLE CUSTOMER PREDICTION
    # ====================================================================
    
    with tab_calc:
        col_form, col_result = st.columns([3, 2])
        
        with col_form:
            with st.form(key="prediction_form"):
                st.subheader("📋 Customer Details")
                
                demo_col, fin_col = st.columns(2)
                
                with demo_col:
                    st.markdown("**👤 Demographics**")
                    age = st.slider("Age", 18, 92, 35,
                                    help="Older customers (55+) tend to have different churn patterns. Younger customers (<30) are often new adopters.")
                    gender = st.selectbox("Gender", ["Male", "Female"],
                                          help="Gender can influence banking preferences and engagement patterns.")
                    geography = st.selectbox("Geography", ["France", "Spain", "Germany"],
                                             help="Geographic location may correlate with regional economic factors and banking habits.")
                    credit_score = st.slider("Credit Score", 350, 850, 600,
                                             help="Higher credit scores (700+) indicate better financial health. Scores below 500 may signal risk.")
                
                with fin_col:
                    st.markdown("**💰 Financial**")
                    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5,
                                       help="Longer tenure (7+ years) suggests loyalty. New customers (<2 years) are more likely to churn.")
                    balance = st.number_input("Account Balance ($)", 0, 250000, 75000, step=1000,
                                              help="Customers with very low or zero balances may be disengaged. High balances indicate strong relationship.")
                    estimated_salary = st.number_input("Estimated Annual Salary ($)", 10000, 200000, 100000, step=5000,
                                                       help="Salary relative to balance helps gauge financial health. High salary + low balance = potential churn risk.")
                
                st.markdown("---")
                prod_col, engage_col = st.columns(2)
                
                with prod_col:
                    st.markdown("**🏦 Products**")
                    num_products = st.slider("Number of Bank Products", 1, 4, 2,
                                             help="Customers with 3+ products (credit card, mortgage, savings, etc.) are more engaged and less likely to churn.")
                    has_credit_card = st.checkbox("Has Credit Card?", value=True,
                                                  help="Credit card holders tend to have deeper engagement with the bank.")
                
                with engage_col:
                    st.markdown("**🔗 Engagement**")
                    is_active = st.checkbox("Active Member?", value=True,
                                            help="Active members (recent transactions, logins) are significantly less likely to churn. This is a top churn driver.")
                
                st.markdown("---")
                
                col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
                with col_btn2:
                    submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary", use_container_width=True)
        
        # Results panel - shown after submission
        with col_result:
            st.subheader("🎯 Risk Assessment")
            
            if submitted:
                # Process prediction
                input_data, churn_probability, churn_prediction, risk_label, risk_class = \
                    process_single_prediction(
                        age, gender, geography, credit_score, tenure, balance,
                        estimated_salary, num_products, has_credit_card, is_active,
                        model, scaler, feature_names
                    )
                
                # Store in session state
                st.session_state.last_inputs = input_data
                
                entry = {
                    'timestamp': pd.Timestamp.now(),
                    'age': age, 'gender': gender, 'geography': geography,
                    'credit_score': credit_score, 'tenure': tenure,
                    'balance': balance, 'salary': estimated_salary,
                    'products': num_products, 'credit_card': has_credit_card,
                    'active': is_active,
                    'probability': churn_probability,
                    'prediction': int(churn_prediction),
                    'risk_label': risk_label
                }
                st.session_state.prediction_history.append(entry)
                
                # Display result
                st.markdown(f"<h2 class='{risk_class}'>{risk_label}</h2>", unsafe_allow_html=True)
                st.metric("Churn Probability", f"{churn_probability*100:.2f}%",
                          help="Probability that this customer will churn (0-100%)")
                
                # Gauge
                fig_gauge = create_gauge_chart(churn_probability)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Key drivers indicator
                st.markdown("**🔑 Key Risk Drivers:**")
                drivers = []
                if is_active == 0:
                    drivers.append("❌ Inactive member")
                if num_products <= 1:
                    drivers.append("⚠️ Only 1 product")
                if tenure < 2:
                    drivers.append("⚠️ New customer (<2 yrs)")
                if age >= 55:
                    drivers.append("⚠️ Senior age group")
                if balance < 1000:
                    drivers.append("⚠️ Low/zero balance")
                
                for d in drivers:
                    st.markdown(f"- {d}")
                if not drivers:
                    st.markdown("- ✅ No major risk factors detected")
                
                # Recommendation
                st.markdown("---")
                st.subheader("💡 Recommended Actions")
                
                if churn_probability >= 0.7:
                    st.error(
                        """
                        **🚨 URGENT ACTION REQUIRED**
                        - Assign dedicated account manager
                        - Offer exclusive retention incentives
                        - Schedule immediate customer check-in
                        - Analyze product usage patterns
                        - Consider personalized loyalty program
                        """
                    )
                elif churn_probability >= 0.4:
                    st.warning(
                        """
                        **⚠️ PROACTIVE ENGAGEMENT NEEDED**
                        - Monitor account activity closely
                        - Send targeted engagement campaigns
                        - Offer premium features/discounts
                        - Gather feedback on satisfaction
                        - Suggest complementary products
                        """
                    )
                else:
                    st.success(
                        """
                        **✅ MAINTAIN RELATIONSHIP**
                        - Continue standard customer service
                        - Occasional cross-sell opportunities
                        - Annual satisfaction surveys
                        - Loyalty rewards program
                        - Regular product updates
                        """
                    )
                
                # What-If analysis
                st.markdown("---")
                st.subheader("🔄 What-If Analysis")
                
                what_if_active = st.checkbox("🔀 Compare: What if engagement changes?", key="what_if_toggle")
                
                if what_if_active:
                    st.markdown("**See how changing engagement affects churn risk:**")
                    
                    what_col1, what_col2 = st.columns(2)
                    
                    with what_col1:
                        st.markdown("**📌 Current Profile**")
                        st.markdown(f"- Active Member: {'Yes' if is_active else 'No'}")
                        st.markdown(f"- Products: {num_products}")
                    
                    with what_col2:
                        st.markdown("**🔄 What If...**")
                        what_active = not is_active
                        st.markdown(f"- Active Member: {'Yes' if what_active else 'No'}")
                    
                    # Simulate what-if
                    what_input_data = build_input_dict(
                        age, gender, geography, credit_score, tenure, balance,
                        estimated_salary, num_products, has_credit_card, what_active
                    )
                    what_features = prepare_input_features(what_input_data, scaler, feature_names)
                    what_prob, what_pred = predict_churn(what_features, model)
                    
                    delta = (what_prob - churn_probability) * 100
                    delta_str = f"{delta:+.1f}%"
                    delta_class = "comparison-delta-positive" if delta > 0 else "comparison-delta-negative"
                    
                    st.markdown(f"""
                    <div class="history-card">
                        <strong>Result:</strong><br>
                        Current: <strong>{churn_probability*100:.1f}%</strong> → 
                        Scenario: <strong>{what_prob*100:.1f}%</strong>
                        <span class="{delta_class}">({delta_str})</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mini comparison gauge
                    fig_compare = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Current", "What-If Scenario"),
                        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]]
                    )
                    
                    fig_compare.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=churn_probability * 100,
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': get_risk_color(churn_probability)}}
                    ), row=1, col=1)
                    
                    fig_compare.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=what_prob * 100,
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': get_risk_color(what_prob)}}
                    ), row=1, col=2)
                    
                    fig_compare.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_compare, use_container_width=True)
            else:
                # Empty state
                st.info("👈 Enter customer details and click **Predict Churn Risk** to see results here.")
                
                # Show placeholder gauge
                fig_placeholder = create_gauge_chart(0.5)
                fig_placeholder.update_traces(gauge_bar_color="rgba(200,200,200,0.3)")
                st.plotly_chart(fig_placeholder, use_container_width=True)
    
    # ====================================================================
    # TAB 2: BATCH UPLOAD
    # ====================================================================
    
    with tab_batch:
        st.subheader("📂 Batch Customer Prediction")
        st.markdown("Upload a CSV file with multiple customers to get bulk churn predictions.")
        
        st.info(
            """
            **CSV Format Requirements:**
            The CSV should contain these columns (case-sensitive):
            - `Age`, `CreditScore`, `Tenure`, `Balance`, `NumOfProducts`
            - `HasCrCard` (0/1), `IsActiveMember` (0/1), `EstimatedSalary`
            - `Geography` (France/Spain/Germany), `Gender` (Male/Female)
            
            Or upload any CSV and we'll try to map columns automatically.
            """
        )
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with customer data. Max file size: 50MB"
        )
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_batch)} customer records")
                
                with st.expander("👁️ Preview Uploaded Data", expanded=False):
                    st.dataframe(df_batch.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 of {len(df_batch)} rows")
                
                # Try to map columns
                col_mapping = {}
                col_lower = {c.lower(): c for c in df_batch.columns}
                
                expected_map = {
                    'age': 'Age', 'credit_score': 'CreditScore', 'credit score': 'CreditScore',
                    'creditscore': 'CreditScore', 'tenure': 'Tenure', 'balance': 'Balance',
                    'numofproducts': 'NumOfProducts', 'num_of_products': 'NumOfProducts',
                    'numberofproducts': 'NumOfProducts', 'has credit card': 'HasCrCard',
                    'hascrcard': 'HasCrCard', 'has_credit_card': 'HasCrCard',
                    'creditcard': 'HasCrCard', 'isactivemember': 'IsActiveMember',
                    'is_active_member': 'IsActiveMember', 'active member': 'IsActiveMember',
                    'active': 'IsActiveMember', 'estimatedsalary': 'EstimatedSalary',
                    'estimated_salary': 'EstimatedSalary', 'salary': 'EstimatedSalary',
                    'geography': 'Geography', 'country': 'Geography', 'gender': 'Gender'
                }
                
                for pattern, target in expected_map.items():
                    if pattern in col_lower:
                        col_mapping[target] = col_lower[pattern]
                
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner(f"Predicting churn for {len(df_batch)} customers..."):
                        results_list = []
                        missing_cols = []
                        
                        for idx, row in df_batch.iterrows():
                            try:
                                age_val = int(row.get(col_mapping.get('Age', 'Age'), 35))
                                gender_val = str(row.get(col_mapping.get('Gender', 'Gender'), 'Male'))
                                geography_val = str(row.get(col_mapping.get('Geography', 'Geography'), 'France'))
                                credit_score_val = float(row.get(col_mapping.get('CreditScore', 'CreditScore'), 600))
                                tenure_val = float(row.get(col_mapping.get('Tenure', 'Tenure'), 5))
                                balance_val = float(row.get(col_mapping.get('Balance', 'Balance'), 75000))
                                salary_val = float(row.get(col_mapping.get('EstimatedSalary', 'EstimatedSalary'), 100000))
                                products_val = int(row.get(col_mapping.get('NumOfProducts', 'NumOfProducts'), 2))
                                card_val = int(row.get(col_mapping.get('HasCrCard', 'HasCrCard'), 1))
                                active_val = int(row.get(col_mapping.get('IsActiveMember', 'IsActiveMember'), 1))
                            except Exception:
                                missing_cols.append(idx)
                                continue
                            
                            _, prob, pred, risk_label, _ = process_single_prediction(
                                age_val, gender_val, geography_val, credit_score_val,
                                tenure_val, balance_val, salary_val, products_val,
                                bool(card_val), bool(active_val), model, scaler, feature_names
                            )
                            
                            results_list.append({
                                'Row': idx + 1,
                                'Churn_Probability': round(prob, 4),
                                'Prediction': 'Will Churn' if pred == 1 else 'Will Stay',
                                'Risk_Category': risk_label
                            })
                        
                        if results_list:
                            df_results = pd.DataFrame(results_list)
                            
                            st.markdown("---")
                            st.subheader("📊 Batch Prediction Results")
                            
                            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                            n_high = len(df_results[df_results['Risk_Category'].str.contains('HIGH')])
                            n_med = len(df_results[df_results['Risk_Category'].str.contains('MEDIUM')])
                            n_low = len(df_results[df_results['Risk_Category'].str.contains('LOW')])
                            avg_risk = df_results['Churn_Probability'].mean()
                            
                            with col_stats1:
                                st.metric("🔴 High Risk", n_high,
                                          help="Customers requiring urgent retention action")
                            with col_stats2:
                                st.metric("🟠 Medium Risk", n_med,
                                          help="Customers needing proactive engagement")
                            with col_stats3:
                                st.metric("🟢 Low Risk", n_low,
                                          help="Customers with stable relationship")
                            with col_stats4:
                                st.metric("📊 Avg Risk", f"{avg_risk*100:.1f}%",
                                          help="Average churn probability across all customers")
                            
                            # Results table (scrollable)
                            st.markdown('<div class="batch-results">', unsafe_allow_html=True)
                            st.dataframe(df_results, use_container_width=True, hide_index=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download button
                            csv_buffer = io.BytesIO()
                            df_results.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_buffer,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Risk distribution chart
                            st.markdown("---")
                            st.subheader("📈 Risk Distribution")
                            
                            risk_counts = df_results['Risk_Category'].value_counts().reset_index()
                            risk_counts.columns = ['Risk', 'Count']
                            
                            fig_dist = px.pie(
                                risk_counts,
                                values='Count',
                                names='Risk',
                                title='Risk Category Distribution',
                                color='Risk',
                                color_discrete_map={
                                    '🔴 HIGH RISK': '#d62728',
                                    '🟠 MEDIUM RISK': '#ff7f0e',
                                    '🟢 LOW RISK': '#2ca02c'
                                },
                                hole=0.4
                            )
                            fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Probability distribution
                            fig_hist = px.histogram(
                                df_results, x='Churn_Probability',
                                nbins=20, title='Churn Probability Distribution',
                                labels={'Churn_Probability': 'Churn Probability'},
                                color_discrete_sequence=['#1f77b4']
                            )
                            fig_hist.add_vline(x=0.4, line_dash="dash", line_color="#ff7f0e", 
                                               annotation_text="Medium Risk Threshold")
                            fig_hist.add_vline(x=0.7, line_dash="dash", line_color="#d62728",
                                               annotation_text="High Risk Threshold")
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        if missing_cols:
                            st.warning(f"⚠️ {len(missing_cols)} rows could not be processed due to missing/wrong data. Rows: {missing_cols[:10]}")
                            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct format. Check the format requirements above.")
    
    # ====================================================================
    # TAB 3: PREDICTION HISTORY
    # ====================================================================
    
    with tab_history:
        st.subheader("📜 Session Prediction History")
        
        if len(st.session_state.prediction_history) == 0:
            st.info("No predictions made yet. Use the **Single Customer** tab to make predictions.")
        else:
            history = st.session_state.prediction_history
            st.success(f"📊 {len(history)} prediction(s) recorded in this session")
            
            # Summary stats
            hist_col1, hist_col2, hist_col3 = st.columns(3)
            with hist_col1:
                avg_prob = np.mean([h['probability'] for h in history])
                st.metric("📊 Avg Predicted Risk", f"{avg_prob*100:.1f}%",
                          help="Average churn probability across all predictions in this session")
            with hist_col2:
                high_count = sum(1 for h in history if h['probability'] >= 0.7)
                st.metric("🔴 High Risk Cases", high_count,
                          help="Number of customers predicted as HIGH risk")
            with hist_col3:
                st.metric("🔄 Total Predictions", len(history),
                          help="Total number of predictions made in this session")
            
            # Show history table
            df_history = pd.DataFrame([{
                'Time': h['timestamp'].strftime('%H:%M:%S'),
                'Age': h['age'],
                'Gender': h['gender'],
                'Geography': h['geography'],
                'Balance': f"${h['balance']:,}",
                'Active': 'Yes' if h['active'] else 'No',
                'Products': h['products'],
                'Risk': h['risk_label']
            } for h in reversed(history)])
            
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            # Chart: Risk trend over time
            if len(history) >= 2:
                st.markdown("---")
                st.subheader("📈 Risk Trend")
                
                df_trend = pd.DataFrame([{
                    'Prediction': i + 1,
                    'Probability': h['probability'] * 100,
                    'Risk': h['risk_label']
                } for i, h in enumerate(history)])
                
                fig_trend = px.line(
                    df_trend, x='Prediction', y='Probability',
                    title='Churn Probability Trend (Session)',
                    markers=True,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_trend.add_hline(y=70, line_dash="dash", line_color="#d62728",
                                    annotation_text="High Risk")
                fig_trend.add_hline(y=40, line_dash="dash", line_color="#ff7f0e",
                                    annotation_text="Medium Risk")
                fig_trend.update_layout(yaxis_title="Churn Probability (%)")
                
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.prediction_history = []
                st.rerun()

def render_model_performance_page(metadata, feature_importance_df, feature_names):
    """Page 3: Detailed model performance"""
    st.title("📈 Detailed Model Performance Analysis")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🔝 Feature Importance", "📋 Model Details"])
    
    with tab1:
        st.subheader("Performance Metrics Comparison")
        
        metrics_list = []
        models_list = []
        
        for model_name, metrics in metadata['model_metrics'].items():
            models_list.append(model_name)
            metrics_list.append(metrics)
        
        metrics_names = ['test_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        display_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Radar chart
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, model_name in enumerate(models_list):
            values = [metrics_list[idx][m] for m in metrics_names] + [metrics_list[idx][metrics_names[0]]]
            theta = display_names + [display_names[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill='toself',
                name=model_name,
                marker=dict(color=colors[idx % len(colors)]),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=True),
                angularaxis=dict(showticklabels=True)
            ),
            showlegend=True,
            height=550,
            margin=dict(l=80, r=80, t=30, b=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart comparison
        st.subheader("📊 Metric Comparison by Model")
        fig_bar = go.Figure()
        
        for idx, model_name in enumerate(models_list):
            values = [metrics_list[idx][m] for m in metrics_names]
            fig_bar.add_trace(go.Bar(
                name=model_name,
                x=display_names,
                y=values,
                marker_color=colors[idx % len(colors)]
            ))
        
        fig_bar.update_layout(
            barmode='group',
            height=400,
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance Analysis")
        
        if feature_importance_df is not None:
            st.write(f"**Total Features: {len(feature_importance_df)}**")
            
            # Top 20 bar chart
            top_20 = feature_importance_df.head(20)
            fig = px.bar(
                top_20,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 20 Most Important Features',
                labels={'importance': 'Importance Score', 'feature': 'Feature Name'},
                color='importance',
                color_continuous_scale='Viridis',
                text='importance'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(height=650, margin=dict(l=10, r=40, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            with st.expander("View All Features"):
                st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Best Model**")
            st.info(f"Model: {metadata['best_model']}")
            st.write(f"**Timestamp**: {metadata['timestamp']}")
        
        with col2:
            st.write("**Model Specifications**")
            st.info(
                f"""
                - **Algorithm**: {metadata['best_model']}
                - **Training Strategy**: Stratified Train-Test Split (80-20)
                - **Feature Scaling**: StandardScaler
                - **Features**: {len(feature_names)} total features
                """
            )
        
        st.write("**Model Metrics**")
        best_metrics = metadata['model_metrics'][metadata['best_model']]
        metric_cols = st.columns(5)
        metric_cols[0].metric("Accuracy", f"{best_metrics['test_accuracy']:.4f}",
                              help="Percentage of correct predictions (both churn and non-churn)")
        metric_cols[1].metric("Precision", f"{best_metrics['precision']:.4f}",
                              help="Of predicted churners, how many actually churn? (reduces false alarms)")
        metric_cols[2].metric("Recall", f"{best_metrics['recall']:.4f}",
                              help="Of actual churners, how many does the model catch?")
        metric_cols[3].metric("F1-Score", f"{best_metrics['f1_score']:.4f}",
                              help="Harmonic mean of precision and recall (balanced metric)")
        metric_cols[4].metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}",
                              help="Model's ability to distinguish churners from non-churners (0.5=random, 1.0=perfect)")

def render_help_page():
    """Page 4: Help & Documentation"""
    st.title("❓ Help & Documentation")
    st.markdown("---")
    
    st.subheader("📖 How to Use This Application")
    
    with st.expander("🎯 Risk Calculator", expanded=True):
        st.write(
            """
            **How to use the Risk Calculator:**
            
            1. Navigate to **"🎯 Risk Calculator"** tab
            2. Select **"Single Customer"** tab
            3. Enter customer information in the form:
               - Demographics (age, gender, location)
               - Financial details (salary, balance)
               - Product usage (number of products, credit card)
               - Engagement status (active member)
            4. Click **"Predict Churn Risk"** button
            5. Review the risk assessment and recommended actions
            6. Try **What-If Analysis** to see impact of engagement changes
            
            **For bulk analysis:**
            1. Go to **"Batch Upload"** tab
            2. Upload a CSV file with customer data
            3. Click **"Run Batch Prediction"**
            4. Download results as CSV
            
            **Risk Categories:**
            - 🟢 **Low Risk** (0-40%): Maintain relationship
            - 🟠 **Medium Risk** (40-70%): Proactive engagement
            - 🔴 **High Risk** (70-100%): Urgent action required
            """
        )
    
    with st.expander("📊 Dashboard"):
        st.write(
            """
            The dashboard provides:
            - Model performance summary
            - Comparison of all trained models
            - Top 10 churn drivers (features)
            - Key business insights
            """
        )
    
    with st.expander("📈 Model Performance"):
        st.write(
            """
            Detailed analysis includes:
            - Radar chart comparing all models across metrics
            - Bar chart comparison by metric
            - Feature importance visualization (top 20)
            - Model specifications and metadata
            """
        )
    
    with st.expander("📜 Prediction History"):
        st.write(
            """
            The history tab tracks all predictions made during your session:
            - View past predictions in a table
            - See churn probability trend over time
            - Summary statistics of session predictions
            - Clear history when needed
            """
        )
    
    st.markdown("---")
    st.subheader("🔍 Understanding Model Metrics")
    
    metric_info = {
        "Accuracy": "Percentage of correct predictions (both churn and non-churn)",
        "Precision": "Of predicted churners, how many actually churn? (reduces false alarms)",
        "Recall": "Of actual churners, how many does the model catch? (identifies at-risk customers)",
        "F1-Score": "Harmonic mean of precision and recall (balanced metric). This is the primary metric used to select the best model.",
        "ROC-AUC": "Model's ability to distinguish churners from non-churners (0.5=random, 1.0=perfect)"
    }
    
    for metric, explanation in metric_info.items():
        with st.expander(f"📌 {metric}"):
            st.write(explanation)
    
    st.markdown("---")
    st.subheader("🏦 Business Context")
    
    st.info(
        """
        **Why Churn Prediction Matters:**
        - Acquiring new customers costs 5-25x more than retaining existing ones
        - A 5% improvement in customer retention can increase profits by 25-95%
        - Proactive retention saves time and money compared to reactive approaches
        
        **Key Churn Drivers (from analysis):**
        - Product engagement (active members churn less)
        - Tenure (newer customers churn more)
        - Age (older customers sometimes more stable)
        - Product diversity (customers with multiple products stay longer)
        """
    )

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state
    init_session_state()
    
    # Load model artifacts
    with st.spinner("🔄 Loading model artifacts..."):
        model, scaler, feature_names, metadata = load_model_artifacts()
    feature_importance_df = load_feature_importance()
    
    # Sidebar
    with st.sidebar:
        st.title("🏦 Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Select a page:",
            ["📊 Dashboard", "🎯 Risk Calculator", "📈 Model Performance", "❓ Help"],
            index=0,
            help="Navigate between different sections of the application"
        )
        
        st.markdown("---")
        st.markdown("**Model Info**")
        st.caption(f"Best Model: {metadata['best_model']}")
        st.caption(f"Last Updated: {metadata['timestamp'][:10]}")
        
        st.markdown("---")
        st.markdown("**💡 Quick Tips**")
        st.caption("• Use Risk Calculator for single predictions")
        st.caption("• Batch Upload for bulk CSV analysis")
        st.caption("• What-If to compare scenarios")
        st.caption("• History tracks your session")
    
    # Render selected page
    if page == "📊 Dashboard":
        render_dashboard_page(metadata, feature_importance_df)
    elif page == "🎯 Risk Calculator":
        render_risk_calculator_page(model, scaler, feature_names)
    elif page == "📈 Model Performance":
        render_model_performance_page(metadata, feature_importance_df, feature_names)
    elif page == "❓ Help":
        render_help_page()

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()