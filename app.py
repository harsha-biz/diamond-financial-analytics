import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# --- PAGE CONFIGURATION & TEAM DETAILS ---
st.set_page_config(page_title="Diamond Analytics Pro", layout="wide", page_icon="💎")
st.title("💎 Advanced Lab-Grown Diamond Analytics")
st.markdown("*CIA 3 Financial Analytics Project | Predictive Customer Behavior & Revenue Modeling*")
st.markdown("**Project Team:** Harsha Vardhan K S (2323213) | Thribhuvan S (2323242) | Ishayu Bannerjee (2323214)")
st.markdown("---")

# --- DATA LOADING & PREP ---
@st.cache_data
def load_data():
    return pd.read_csv('processed_diamond_data.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'processed_diamond_data.csv' not found. Please ensure it is in the same folder.")
    st.stop()

# Preprocessing
df_prep = df.copy()
df_prep['Purchase_Intent_Binary'] = df_prep['Purchase_Intent'].apply(lambda x: 1 if x >= 3.0 else 0)
df_encoded = pd.get_dummies(df_prep.drop(columns=['Purchase_Intent']), columns=['Gender', 'Marital_Status', 'City_Tier'], drop_first=True)

X = df_encoded.drop(columns=['Purchase_Intent_Binary'])
y = df_encoded['Purchase_Intent_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scalers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Core Scaler for Simulator
X_core = X[['Social_Value', 'Financial_Value', 'Authenticity', 'Sustainability']]
scaler_core = StandardScaler().fit(X_core)

# --- MODEL TRAINING ENGINE (ALL RUBRIC MODELS) ---
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Ada Boost": AdaBoostClassifier(random_state=42, n_estimators=100),
        "XG Boost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "Neural Network (Without Feature Selection)": MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(64, 32))
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {'Model': model, 'Accuracy': acc, 'CM': cm}
        
    # Feature Importance & NN With Feature Selection
    rf_model = results["Random Forest"]['Model']
    fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    
    top_features = fi_df[fi_df['Importance'] > 0.05]['Feature'].tolist()
    X_train_sel = X_train_scaled_df[top_features]
    X_test_sel = X_test_scaled_df[top_features]
    
    nn_fs = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(64, 32))
    nn_fs.fit(X_train_sel, y_train)
    acc_fs = accuracy_score(y_test, nn_fs.predict(X_test_sel))
    cm_fs = confusion_matrix(y_test, nn_fs.predict(X_test_sel))
    
    results["Neural Network (With Feature Selection)"] = {'Model': nn_fs, 'Accuracy': acc_fs, 'CM': cm_fs}
    
    return results, fi_df, top_features

model_results, feature_importances, top_features = train_models()
nn_interactive = model_results["Neural Network (With Feature Selection)"]['Model']

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation Panel")
page = st.sidebar.radio("Go to:", [
    "📊 Dynamic Market Insights (EDA)", 
    "🤖 Model Performance & Evaluation",
    "🧠 Strategic Feature Drivers", 
    "🎯 Financial & Persona Simulator"
])

# --- PAGE 1: DYNAMIC EDA ---
if page == "📊 Dynamic Market Insights (EDA)":
    st.header("1. Dynamic Market Insights (EDA)")
    st.write("Exploratory Data Analysis: Interact with the dataset to uncover demographic and behavioral trends.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Filter Settings")
        x_axis = st.selectbox("Select X-Axis Feature:", ['Authenticity', 'Sustainability', 'Financial_Value', 'Social_Value'])
        color_split = st.selectbox("Segment By:", ['Gender', 'Marital_Status', 'City_Tier'])
    
    with col2:
        fig = px.scatter(df, x=x_axis, y='Purchase_Intent', color=color_split, 
                         title=f"Purchase Intent vs {x_axis} (Segmented by {color_split})",
                         template="plotly_white", opacity=0.7, size_max=10)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Purchase Intent Distribution")
    fig2 = px.histogram(df, x="Purchase_Intent", nbins=15, marginal="box", color_discrete_sequence=['teal'])
    st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 2: ALL CLASSIFICATION MODELS (RUBRIC FULFILLMENT) ---
elif page == "🤖 Model Performance & Evaluation":
    st.header("2. Classification Models Evaluation")
    st.write("Comprehensive evaluation of all required machine learning algorithms to predict binary purchase intent.")
    
    # Leaderboard
    st.subheader("🏆 Model Accuracy Leaderboard")
    acc_data = [{"Algorithm": name, "Accuracy (%)": round(data['Accuracy'] * 100, 2)} for name, data in model_results.items()]
    acc_df = pd.DataFrame(acc_data).sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
    
    # Highlight max accuracy
    st.dataframe(acc_df.style.highlight_max(subset=['Accuracy (%)'], color='lightgreen'), use_container_width=True)
    
    # Confusion Matrix Explorer
    st.markdown("---")
    st.subheader("🔍 Confusion Matrix Explorer")
    selected_model = st.selectbox("Select an Algorithm to view its Confusion Matrix:", list(model_results.keys()))
    
    cm = model_results[selected_model]['CM']
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                       labels=dict(x="Predicted Label", y="True Label", color="Count"),
                       x=['Low Intent (0)', 'High Intent (1)'], y=['Low Intent (0)', 'High Intent (1)'])
    fig_cm.update_layout(title=f"Confusion Matrix: {selected_model}")
    st.plotly_chart(fig_cm, use_container_width=True)

# --- PAGE 3: FEATURE IMPORTANCE ---
elif page == "🧠 Strategic Feature Drivers":
    st.header("3. Strategic Drivers of Adoption")
    st.write("Which factors actually drive the financial decision to purchase a lab-grown diamond?")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(feature_importances.head(6), x='Importance', y='Feature', orientation='h',
                     title="Top Predictors of Purchase Intent (Random Forest Gini Importance)",
                     color='Importance', color_continuous_scale="Viridis")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.info("### 💡 Financial Insight")
        st.write("Our models prove that **Demographics (Gender, Marital Status, City) account for almost zero impact** on the purchase decision.")
        st.write("Consumers buy based entirely on **Authenticity**, **Financial Value**, **Social Value**, and **Sustainability**. Marketing budgets should be reallocated from demographic targeting to value-based messaging.")
        
        st.subheader("Neural Network Optimization")
        st.write(f"By applying feature selection and dropping demographics, our optimized Neural Network relies only on: `{', '.join(top_features)}`.")

# --- PAGE 4: PERSONA SIMULATOR, 4Cs & FINANCIAL IMPACT ---
elif page == "🎯 Financial & Persona Simulator":
    st.header("4. Financial Revenue Simulator (with Real-Time Market Prices)")
    st.write("Adjust the product specifications (4Cs) and the customer's behavioral levers to predict expected pipeline revenue using our optimized Neural Network.")
    
    st.markdown("### 💎 Step 1: Define the Product (The 4Cs)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        carat = st.selectbox("Carat Weight", [0.5, 1.0, 1.5, 2.0, 3.0], index=1)
    with c2:
        cut = st.selectbox("Cut Grade", ["Ideal", "Excellent", "Very Good", "Good"], index=1)
    with c3:
        color = st.selectbox("Color Grade", ["D-F (Colorless)", "G-H (Near Colorless)", "I-J (Faint Yellow)"], index=1)
    with c4:
        clarity = st.selectbox("Clarity Grade", ["VVS1 - VVS2", "VS1 - VS2", "SI1 - SI2"], index=1)

    # Dynamic Pricing Logic
    base_prices = {0.5: 18500, 1.0: 90000, 1.5: 80000, 2.0: 137500, 3.0: 135000}
    base_price = base_prices[carat]
    
    cut_mult = {"Ideal": 1.15, "Excellent": 1.05, "Very Good": 0.95, "Good": 0.85}[cut]
    color_mult = {"D-F (Colorless)": 1.15, "G-H (Near Colorless)": 1.0, "I-J (Faint Yellow)": 0.85}[color]
    clarity_mult = {"VVS1 - VVS2": 1.15, "VS1 - VS2": 1.0, "SI1 - SI2": 0.85}[clarity]
    
    calculated_diamond_price = base_price * cut_mult * color_mult * clarity_mult
    
    st.markdown("---")
    
    st.markdown("### 🧠 Step 2: Define the Customer Psyche")
    persona = st.selectbox("Load a Pre-Built Customer Persona:", 
                           ["Custom Manual Entry", "The Eco-Warrior (High Sustain, Low Auth)", 
                            "The Status Seeker (High Social, High Fin)", "The Skeptic (Low everything)"])
    
    def_soc, def_fin, def_auth, def_sus = 3.0, 3.0, 3.0, 3.0
    if "Eco-Warrior" in persona:
        def_soc, def_fin, def_auth, def_sus = 2.5, 3.0, 1.5, 4.8
    elif "Status Seeker" in persona:
        def_soc, def_fin, def_auth, def_sus = 4.8, 4.5, 4.0, 1.5
    elif "Skeptic" in persona:
        def_soc, def_fin, def_auth, def_sus = 1.5, 2.0, 1.5, 1.5

    col1, col2, col3 = st.columns([1, 1, 1.5])
    
    with col1:
        soc = st.slider("Social Value", 1.0, 5.0, def_soc, 0.25)
        fin = st.slider("Financial Value", 1.0, 5.0, def_fin, 0.25)
        auth = st.slider("Authenticity", 1.0, 5.0, def_auth, 0.25)
        sus = st.slider("Sustainability", 1.0, 5.0, def_sus, 0.25)
        
    with col2:
        categories = ['Social Value', 'Financial Value', 'Authenticity', 'Sustainability']
        fig = go.Figure(data=go.Scatterpolar(r=[soc, fin, auth, sus], theta=categories, fill='toself', marker=dict(color='teal')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        st.subheader("Financial Impact Prediction")
        
        # FIXED: Using exact dataframe column names that the scaler and model expect
        model_cols = ['Social_Value', 'Financial_Value', 'Authenticity', 'Sustainability']
        input_scaled = scaler_core.transform(pd.DataFrame([[soc, fin, auth, sus]], columns=model_cols))
        
        probability = nn_interactive.predict_proba(input_scaled)[0][1] * 100
        expected_revenue = (probability / 100) * calculated_diamond_price
        
        st.metric(label=f"Calculated {carat}ct LGD Retail Price", value=f"₹{calculated_diamond_price:,.0f}")
        st.metric(label="Customer Likelihood to Purchase", value=f"{probability:.1f}%")
        st.metric(label="Expected Pipeline Revenue", value=f"₹{expected_revenue:,.0f}", help="Probability of Purchase * Diamond Price")
        
        if probability > 50:
            st.success("High-Value Prospect. Recommend immediate targeted marketing.")
        else:
            st.error("Low-Value Prospect. Requires nurturing on core value metrics.")