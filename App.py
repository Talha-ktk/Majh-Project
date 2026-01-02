import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc, roc_auc_score)
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Student Impact Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9;
    }
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        text-align: center;
    }
    .prediction-card-pass {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-left: 8px solid #10b981;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
        margin: 15px 0;
    }
    .prediction-card-fail {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border-left: 8px solid #ef4444;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.4);
        margin: 15px 0;
    }
    h1 {
        color: #10b981;
        font-weight: 800;
    }
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        padding: 12px 32px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_resource
def generate_dataset(n_samples=1000):
    np.random.seed(42)
    
    age = np.random.randint(15, 31, n_samples)
    gender = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.50, 0.05])
    grade_level = np.random.choice(['1st Year', '2nd Year', '3rd Year', '4th Year'], n_samples)
    
    study_hours = np.random.normal(4.5, 2, n_samples).clip(0.5, 10)
    exam_score = np.random.normal(65, 20, n_samples).clip(10, 100)
    assignment_avg = np.random.normal(63, 22, n_samples).clip(10, 100)
    attendance = np.random.normal(78, 15, n_samples).clip(20, 100)
    concept_score = np.random.randint(1, 11, n_samples)
    
    uses_ai = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    ai_time = np.random.exponential(45, n_samples) * uses_ai
    ai_dependency = np.random.randint(1, 11, n_samples)
    ai_content = np.random.uniform(0, 100, n_samples) * uses_ai
    ai_tools = np.random.choice(['ChatGPT', 'Gemini', 'Copilot', 'Claude', 'None'], n_samples)
    ai_purpose = np.random.choice(['Exam Prep', 'Assignment Help', 'Concept Learning'], n_samples)
    
    sleep_hours = np.random.normal(7, 1.5, n_samples).clip(3, 10)
    social_media = np.random.exponential(2.5, n_samples).clip(0, 10)
    class_participation = np.random.randint(1, 11, n_samples)
    
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'grade_level': grade_level,
        'study_hours_per_day': study_hours,
        'uses_ai': uses_ai,
        'ai_usage_time_minutes': ai_time,
        'ai_tools_used': ai_tools,
        'ai_usage_purpose': ai_purpose,
        'ai_dependency_score': ai_dependency,
        'ai_generated_content_percentage': ai_content,
        'last_exam_score': exam_score,
        'assignment_scores_avg': assignment_avg,
        'attendance_percentage': attendance,
        'concept_understanding_score': concept_score,
        'sleep_hours': sleep_hours,
        'social_media_hours': social_media,
        'class_participation_score': class_participation
    })
    
    df['passed'] = (
        (df['last_exam_score'] > 40) & 
        (df['assignment_scores_avg'] > 35) &
        (df['attendance_percentage'] > 60) &
        (df['study_hours_per_day'] > 1.5)
    ).astype(int)
    
    noise_idx = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
    df.loc[noise_idx, 'passed'] = 1 - df.loc[noise_idx, 'passed']
    
    return df

@st.cache_resource
def train_model():
    df = generate_dataset(1000)
    
    X = df.drop(['passed'], axis=1)
    y = df['passed']
    
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    X = X.fillna(X.mean(numeric_only=True))
    
    scaler = StandardScaler()
    feature_names = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return model, feature_names, label_encoders, metrics, test_data, scaler

model, feature_names, label_encoders, metrics, test_data, scaler = train_model()

def prepare_input(input_df):
    X_input = input_df.copy()
    
    for col in label_encoders:
        if col in X_input.columns:
            try:
                X_input[col] = label_encoders[col].transform(X_input[col].astype(str))
            except:
                X_input[col] = 0
    
    X_input = X_input.fillna(0)
    
    for feat in feature_names:
        if feat not in X_input.columns:
            X_input[feat] = 0
    
    X_input = X_input[feature_names]
    X_scaled = scaler.transform(X_input)
    
    return pd.DataFrame(X_scaled, columns=feature_names)

def get_top_features(n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    return [(feature_names[i], importances[i]) for i in indices]

st.markdown('<div class="main-header"><h1>AI Student Impact Predictor</h1><p>Advanced ML Model for Student Performance</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Analytics", "Features", "About"])

with tab1:
    st.header("Student Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Academic")
        exam = st.slider("Exam Score", 0, 100, 72)
        assignment = st.slider("Assignment Avg", 0, 100, 68)
        concept = st.slider("Concept (1-10)", 1, 10, 6)
        attendance = st.slider("Attendance %", 0, 100, 82)
    
    with col2:
        st.subheader("AI Usage")
        ai_tool = st.selectbox("AI Tool", ['ChatGPT', 'Gemini', 'Copilot', 'Claude', 'None'])
        ai_time = st.number_input("AI Time (Min/Day)", 0, 500, 45)
        ai_content = st.slider("AI Content %", 0, 100, 25)
        ai_depend = st.slider("AI Dependency", 1, 10, 4)
    
    with col3:
        st.subheader("Lifestyle")
        age = st.number_input("Age", 15, 30, 21)
        study = st.number_input("Study Hours", 0.0, 12.0, 4.2)
        sleep = st.number_input("Sleep Hours", 3.0, 12.0, 7.5)
        social = st.number_input("Social Media Hours", 0.0, 10.0, 2.5)
    
    st.divider()
    
    grade = st.selectbox("Grade Level", ['1st Year', '2nd Year', '3rd Year', '4th Year'])
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    ai_purpose = st.selectbox("AI Purpose", ['Exam Prep', 'Assignment Help', 'Concept Learning'])
    class_part = st.slider("Class Participation", 1, 10, 6)
    
    st.divider()
    
    uses_ai = 1 if ai_tool != 'None' else 0
    
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'grade_level': [grade],
        'study_hours_per_day': [study],
        'uses_ai': [uses_ai],
        'ai_usage_time_minutes': [ai_time],
        'ai_tools_used': [ai_tool],
        'ai_usage_purpose': [ai_purpose],
        'ai_dependency_score': [ai_depend],
        'ai_generated_content_percentage': [ai_content],
        'last_exam_score': [exam],
        'assignment_scores_avg': [assignment],
        'attendance_percentage': [attendance],
        'concept_understanding_score': [concept],
        'sleep_hours': [sleep],
        'social_media_hours': [social],
        'class_participation_score': [class_part]
    })
    
    if st.button("Run Prediction", use_container_width=True):
        prepared = prepare_input(input_data)
        pred = model.predict(prepared)[0]
        proba = model.predict_proba(prepared)[0]
        conf = max(proba)
        
        st.divider()
        
        if pred == 1:
            st.markdown('<div class="prediction-card-pass"><h2 style="color: #10b981;">PASS - {:.1%} Confidence</h2></div>'.format(conf), unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-card-fail"><h2 style="color: #ef4444;">FAIL - {:.1%} Risk</h2></div>'.format(1-conf), unsafe_allow_html=True)
        
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            fig = go.Figure(data=[go.Bar(x=['FAIL', 'PASS'], y=[proba[0], proba[1]], marker_color=['#ef4444', '#10b981'])])
            fig.update_layout(title="Probabilities", template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_prob2:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=conf*100, title="Confidence", gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#10b981"))))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Factors")
        top = get_top_features(4)
        cols = st.columns(4)
        for i, (feat, imp) in enumerate(top):
            cols[i].metric(feat, f"{imp:.2%}")

with tab2:
    st.header("Model Performance")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall", f"{metrics['recall']:.2%}")
    m4.metric("F1-Score", f"{metrics['f1']:.2%}")
    m5.metric("AUC", f"{metrics['auc']:.2%}")
    
    st.divider()
    
    col_feat, col_conf = st.columns(2)
    
    with col_feat:
        top_all = get_top_features(12)
        fig = go.Figure(data=[go.Bar(y=[f[0] for f in top_all], x=[f[1] for f in top_all], orientation='h', marker_color=[f[1] for f in top_all], marker_colorscale='Viridis')])
        fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_conf:
        y_test = test_data['y_test']
        y_pred = test_data['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Fail', 'Pass'], y=['Fail', 'Pass'], text=cm, texttemplate='%{text}', colorscale='Blues'))
        fig.update_layout(title="Confusion Matrix", height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    y_test = test_data['y_test']
    y_pred_proba = test_data['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#10b981', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
    fig.update_layout(
        title="ROC Curve", 
        xaxis_title="False Positive Rate", 
        yaxis_title="True Positive Rate", 
        height=400, 
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Feature Analysis")
    
    st.markdown("The model analyzes 17 student features across 4 categories:")
    
    st.divider()
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.subheader("Academic")
        st.write("- Exam Score\n- Assignment Avg\n- Attendance\n- Concept Level\n- Class Part")
    
    with col_b:
        st.subheader("AI Usage")
        st.write("- Tool Selection\n- Usage Time\n- Dependency\n- Content %")
    
    with col_c:
        st.subheader("Demographics")
        st.write("- Age\n- Gender\n- Grade Level")
    
    with col_d:
        st.subheader("Lifestyle")
        st.write("- Study Hours\n- Sleep Hours\n- Social Media\n- Class Part")

with tab4:
    st.header("About")
    
    st.markdown("""
    ### Machine Learning Model
    - **Algorithm:** Random Forest Classifier
    - **Trees:** 200
    - **Features:** 17 metrics
    - **Accuracy:** ~85%
    
    ### Features Analyzed
    The model evaluates academic performance, AI tool usage, 
    demographic data, and lifestyle factors.
    
    ### Disclaimer
    This tool provides predictions based on historical patterns. 
    Always consult with academic advisors for personalized guidance.
    """)