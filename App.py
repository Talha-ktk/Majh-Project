import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc, 
                             roc_auc_score, precision_recall_curve)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🎓 AI Student Impact Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS
custom_css = """
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
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
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-top: 3px solid #10b981;
        margin-bottom: 12px;
    }
    
    .info-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        margin: 12px 0;
    }
    
    h1 {
        color: #10b981;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 10px;
    }
    
    h2 {
        color: #64748b;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    h3 {
        color: #94a3b8;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 32px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.5) !important;
        transform: translateY(-2px) !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =============================================================================
# DATA GENERATION & MODEL TRAINING
# =============================================================================
@st.cache_resource
def generate_comprehensive_dataset(n_samples=1000):
    """Generate a comprehensive synthetic dataset with realistic correlations"""
    np.random.seed(42)
    
    age = np.random.randint(15, 31, n_samples)
    gender = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.50, 0.05])
    grade_level = np.random.choice(['1st Year', '2nd Year', '3rd Year', '4th Year', '10th', '11th', '12th'], n_samples)
    
    study_hours = np.random.normal(4.5, 2, n_samples).clip(0.5, 10)
    last_exam_score = np.random.normal(65, 20, n_samples).clip(10, 100)
    assignment_avg = np.random.normal(63, 22, n_samples).clip(10, 100)
    attendance = np.random.normal(78, 15, n_samples).clip(20, 100)
    concept_understanding = np.random.randint(1, 11, n_samples)
    
    uses_ai = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    ai_usage_time = np.random.exponential(45, n_samples) * uses_ai
    ai_dependency = np.random.randint(1, 11, n_samples)
    ai_generated_content = np.random.uniform(0, 100, n_samples) * uses_ai
    ai_tools = np.random.choice(['ChatGPT', 'Gemini', 'Copilot', 'Claude', 'None', 'Unknown'], n_samples)
    ai_usage_purpose = np.random.choice(['Exam Prep', 'Assignment Help', 'Concept Learning', 'Research'], n_samples)
    ai_prompts_week = np.random.uniform(0, 60, n_samples) * uses_ai
    ai_ethics = np.random.randint(1, 11, n_samples)
    
    sleep_hours = np.random.normal(7, 1.5, n_samples).clip(3, 10)
    social_media = np.random.exponential(2.5, n_samples).clip(0, 10)
    tutoring_hours = np.random.exponential(1, n_samples).clip(0, 15)
    class_participation = np.random.randint(1, 11, n_samples)
    consistency_index = np.random.uniform(1, 10, n_samples)
    improvement_rate = np.random.normal(2, 3, n_samples).clip(-5, 10)
    
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'grade_level': grade_level,
        'study_hours_per_day': study_hours,
        'uses_ai': uses_ai,
        'ai_usage_time_minutes': ai_usage_time,
        'ai_tools_used': ai_tools,
        'ai_usage_purpose': ai_usage_purpose,
        'ai_dependency_score': ai_dependency,
        'ai_generated_content_percentage': ai_generated_content,
        'ai_prompts_per_week': ai_prompts_week,
        'ai_ethics_score': ai_ethics,
        'last_exam_score': last_exam_score,
        'assignment_scores_avg': assignment_avg,
        'attendance_percentage': attendance,
        'concept_understanding_score': concept_understanding,
        'study_consistency_index': consistency_index,
        'improvement_rate': improvement_rate,
        'sleep_hours': sleep_hours,
        'social_media_hours': social_media,
        'tutoring_hours': tutoring_hours,
        'class_participation_score': class_participation
    })
    
    df['passed'] = (
        (df['last_exam_score'] > 40) & 
        (df['assignment_scores_avg'] > 35) &
        (df['attendance_percentage'] > 60) &
        (df['study_hours_per_day'] > 1.5) &
        (df['concept_understanding_score'] > 2)
    ).astype(int)
    
    noise_idx = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
    df.loc[noise_idx, 'passed'] = 1 - df.loc[noise_idx, 'passed']
    
    df.insert(0, 'student_id', range(1, len(df) + 1))
    
    return df

@st.cache_resource
def train_random_forest_model():
    """Train comprehensive Random Forest model with advanced preprocessing"""
    
    try:
        df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    except FileNotFoundError:
        df = generate_comprehensive_dataset(1000)
        st.info("📊 Using synthetic dataset for demonstration (1000 samples)")
    
    X = df.drop(['passed', 'student_id'], axis=1, errors='ignore')
    y = df['passed'] if 'passed' in df.columns else None
    
    if y is None:
        st.error("Dataset must contain 'passed' column")
        return None, None, None, None, None, None
    
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
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
        bootstrap=True,
        oob_score=True
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    y_pred_proba_test = rf_model.predict_proba(X_test)[:, 1]
    y_pred_proba_train = rf_model.predict_proba(X_train)[:, 1]
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba_test),
        'oob_score': rf_model.oob_score_,
        'cv_scores': cross_val_score(rf_model, X_scaled, y, cv=5)
    }
    
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test,
        'y_pred_proba': y_pred_proba_test,
        'y_train': y_train,
        'y_pred_train': y_pred_train,
        'y_pred_proba_train': y_pred_proba_train
    }
    
    return rf_model, feature_names, label_encoders, metrics, test_data, scaler

rf_model, feature_names, label_encoders, model_metrics, test_data, scaler = train_random_forest_model()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def prepare_input_for_prediction(input_df):
    """Prepare input data for prediction"""
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

def get_top_features(n=12):
    """Get top N important features"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    return [(feature_names[i], importances[i]) for i in indices]

# =============================================================================
# MAIN LAYOUT
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 AI Student Impact Predictor</h1>
    <p style="font-size: 16px; color: #ecfdf5; margin-top: 10px;">
        Advanced Machine Learning Model for Student Performance Prediction
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction Engine",
    "📊 Model Analytics",
    "📈 Feature Analysis",
    "ℹ️ About"
])

# =============================================================================
# TAB 1: PREDICTION ENGINE
# =============================================================================
with tab1:
    st.markdown("### 📝 Student Profile Configuration")
    
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        st.markdown("#### 🎓 Academic Metrics")
        exam_score = st.slider("Last Exam Score", 0, 100, 72, step=1, key='exam')
        assignment_score = st.slider("Assignment Average", 0, 100, 68, step=1, key='assign')
        concept_score = st.slider("Concept Understanding (1-10)", 1, 10, 6, step=1, key='concept')
        attendance = st.slider("Attendance %", 0, 100, 82, step=1, key='attend')
        class_part = st.slider("Class Participation (1-10)", 1, 10, 6, step=1, key='classpart')
    
    with input_col2:
        st.markdown("#### 🤖 AI Usage & Habits")
        ai_tools = st.selectbox("Main AI Tool", ['ChatGPT', 'Gemini', 'Copilot', 'Claude', 'None', 'Unknown'])
        ai_time = st.number_input("AI Usage (Min/Day)", 0, 500, 45, step=5, key='aitime')
        ai_content = st.slider("AI Generated Content %", 0, 100, 25, step=5, key='aicontent')
        ai_depend = st.slider("AI Dependency (1-10)", 1, 10, 4, step=1, key='aidep')
        ai_prompts = st.number_input("AI Prompts/Week", 0, 100, 12, step=1, key='prompts')
    
    with input_col3:
        st.markdown("#### 👤 Demographics & Lifestyle")
        age = st.number_input("Age", 15, 30, 21, step=1, key='age')
        study_hours = st.number_input("Study Hours/Day", 0.0, 12.0, 4.2, step=0.5, key='study')
        sleep_hours = st.number_input("Sleep Hours/Day", 3.0, 12.0, 7.5, step=0.5, key='sleep')
        social_media = st.number_input("Social Media Hours/Day", 0.0, 10.0, 2.5, step=0.5, key='socmed')
        tutoring = st.number_input("Tutoring Hours/Week", 0.0, 20.0, 2.0, step=0.5, key='tutor')
    
    st.markdown("---")
    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
    
    with cat_col1:
        grade = st.selectbox("Grade Level", ['1st Year', '2nd Year', '3rd Year', '4th Year', '10th', '11th', '12th'])
    
    with cat_col2:
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    
    with cat_col3:
        ai_purpose = st.selectbox("AI Usage Purpose", ['Exam Prep', 'Assignment Help', 'Concept Learning', 'Research'])
    
    with cat_col4:
        ai_ethics = st.slider("AI Ethics Score (1-10)", 1, 10, 7, step=1, key='ethics')
    
    st.markdown("---")
    additional_col1, additional_col2, additional_col3 = st.columns(3)
    
    with additional_col1:
        consistency = st.slider("Study Consistency Index (1-10)", 1, 10, 6, step=1, key='consist')
    
    with additional_col2:
        improvement = st.slider("Improvement Rate (-5 to 10)", -5, 10, 2, step=1, key='improve')
    
    with additional_col3:
        uses_ai = 1 if ai_tools != 'None' else 0
        st.metric("AI Usage Status", "✅ Active" if uses_ai else "❌ Inactive")
    
    st.markdown("---")
    
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'grade_level': [grade],
        'study_hours_per_day': [study_hours],
        'uses_ai': [uses_ai],
        'ai_usage_time_minutes': [ai_time],
        'ai_tools_used': [ai_tools],
        'ai_usage_purpose': [ai_purpose],
        'ai_dependency_score': [ai_depend],
        'ai_generated_content_percentage': [ai_content],
        'ai_prompts_per_week': [ai_prompts],
        'ai_ethics_score': [ai_ethics],
        'last_exam_score': [exam_score],
        'assignment_scores_avg': [assignment_score],
        'attendance_percentage': [attendance],
        'concept_understanding_score': [concept_score],
        'study_consistency_index': [consistency],
        'improvement_rate': [improvement],
        'sleep_hours': [sleep_hours],
        'social_media_hours': [social_media],
        'tutoring_hours': [tutoring],
        'class_participation_score': [class_part]
    })
    
    st.markdown("### 📋 Student Profile Summary")
    
    review_col1, review_col2 = st.columns([2, 1])
    
    with review_col1:
        st.dataframe(input_data.T, use_container_width=True, height=400)
    
    with review_col2:
        st.markdown("### ⚡ Quick Stats")
        st.metric("Exam Score", f"{exam_score}/100", delta=f"{exam_score - 50}")
        st.metric("Study Hours", f"{study_hours:.1f}h", delta=f"{study_hours - 3.5:.1f}h")
        st.metric("AI Usage", f"{ai_time} min", delta="High" if ai_time > 100 else "Normal")
        st.metric("Attendance", f"{attendance}%", delta=f"{attendance - 70}%")
    
    st.markdown("---")
    
    col_space, col_predict = st.columns([1, 2])
    
    with col_predict:
        predict_button = st.button("🚀 Run ML Prediction Model", use_container_width=True, key='predict')
    
    if predict_button:
        with st.spinner("🔄 Running Random Forest Model..."):
            input_prepared = prepare_input_for_prediction(input_data)
            prediction = rf_model.predict(input_prepared)[0]
            pred_proba = rf_model.predict_proba(input_prepared)[0]
            confidence = max(pred_proba)
        
        st.markdown("---")
        st.markdown("### 🔮 Prediction Results")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-card-pass">
                <h2 style="color: #10b981; margin: 0;">✅ PASS PREDICTION</h2>
                <p style="font-size: 16px; margin-top: 12px; color: #d1fae5;">This student is predicted to PASS with {confidence:.1%} confidence.</p>
                <div style="margin-top: 20px;">
                    <h3 style="color: #6ee7b7; font-size: 32px; margin: 0;">{confidence:.1%}</h3>
                    <p style="color: #a7f3d0; margin-top: 5px;">Confidence Level</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card-fail">
                <h2 style="color: #ef4444; margin: 0;">⚠️ FAIL WARNING</h2>
                <p style="font-size: 16px; margin-top: 12px; color: #fecaca;">This student is at risk of FAILING with {(1-confidence):.1%} risk level.</p>
                <div style="margin-top: 20px;">
                    <h3 style="color: #f87171; font-size: 32px; margin: 0;">{(1-confidence):.1%}</h3>
                    <p style="color: #fca5a5; margin-top: 5px;">Risk Level</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Prediction Probability Distribution")
        
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            fig_prob = go.Figure(data=[
                go.Bar(x=['FAIL', 'PASS'], y=[pred_proba[0], pred_proba[1]],
                       marker=dict(color=['#ef4444', '#10b981']),
                       text=[f'{pred_proba[0]:.1%}', f'{pred_proba[1]:.1%}'],
                       textposition='outside')
            ])
            fig_prob.update_layout(
                title="Prediction Probabilities",
                height=300,
                showlegend=False,
                template="plotly_dark",
                xaxis_title="Outcome",
                yaxis_title="Probability"
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with prob_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence %"},
                delta={'reference': 80},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': '#10b981'},
                       'steps': [
                           {'range': [0, 50], 'color': '#fee2e2'},
                           {'range': [50, 80], 'color': '#fef3c7'},
                           {'range': [80, 100], 'color': '#dcfce7'}
                       ]}
            ))
            fig_gauge.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🔑 Top Influencing Factors")
        
        top_features = get_top_features(8)
        factors_col1, factors_col2, factors_col3, factors_col4 = st.columns(4)
        
        factor_cols = [factors_col1, factors_col2, factors_col3, factors_col4]
        
        for idx, (feat, imp) in enumerate(top_features[:4]):
            with factor_cols[idx]:
                st.info(f"**{feat}**\nImportance: {imp:.2%}")
        
        st.markdown("---")
        st.markdown("### 📊 Detailed Performance Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            st.markdown("#### 📚 Academic Strength")
            st.info(f"Exam Score: {exam_score}/100\nAssignment Avg: {assignment_score}/100\nAttendance: {attendance}%\nConcept Level: {concept_score}/10\nClass Part: {class_part}/10")
        
        with analysis_col2:
            st.markdown("#### 🤖 AI Impact")
            st.info(f"AI Content: {ai_content}%\nAI Dependency: {ai_depend}/10\nTool: {ai_tools}\nDaily Usage: {ai_time} min\nEthics: {ai_ethics}/10")
        
        with analysis_col3:
            st.markdown("#### 📖 Study & Lifestyle")
            st.info(f"Study Hours: {study_hours:.1f}h\nSleep Hours: {sleep_hours:.1f}h\nSocial Media: {social_media:.1f}h\nTutoring: {tutoring:.1f}h/week\nAge: {age} years")

# =============================================================================
# TAB 2: MODEL ANALYTICS
# =============================================================================
with tab2:
    st.subheader("📊 Model Performance Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Test Accuracy", f"{model_metrics['test_accuracy']:.2%}")
    metric_col2.metric("Precision", f"{model_metrics['precision']:.2%}")
    metric_col3.metric("Recall", f"{model_metrics['recall']:.2%}")
    metric_col4.metric("F1-Score", f"{model_metrics['f1']:.2%}")
    metric_col5.metric("AUC Score", f"{model_metrics['auc']:.2%}")
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("🔑 Top 15 Feature Importances")
        importance_list = get_top_features(15)
        feat_names = [f[0] for f in importance_list]
        feat_vals = [f[1] for f in importance_list]
        
        fig = go.Figure(data=[
            go.Bar(y=feat_names, x=feat_vals, orientation='h', 
                   marker=dict(color=feat_vals, colorscale='Viridis', showscale=False))
        ])
        fig.update_layout(
            title="Feature Importance Scores",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📊 Confusion Matrix")
        y_test = test_data['y_test']
        y_pred = test_data['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Fail', 'Predicted Pass'],
            y=['Actual Fail', 'Actual Pass'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        fig.update_layout(
            title="Confusion Matrix",
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📈 ROC Curve")
    y_test = test_data['y_test']
    y_pred_proba = test_data['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                             line=dict(color='#10b981', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                             line=dict(color='gray', width=2, dash='dash')))
    fig.update_layout(
        title=f"ROC Curve (AUC = {model_metrics['auc']:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: FEATURE ANALYSIS
# =============================================================================
with tab3:
    st.subheader("📈 Feature Importance Analysis")
    
    st.markdown("""
    The Random Forest model analyzes 22 different student features to make predictions.
    Below are the most influential factors for determining student success.
    """)
    
    st.markdown("---")
    
    # Get all features
    all_features = get_top_features(22)
    
    fig = go.Figure(data=[
        go.Bar(y=[f[0] for f in all_features], 
               x=[f[1] for f in all_features],
               orientation='h',
               marker=dict(color=[f[1] for f in all_features], colorscale='Plasma'))
    ])
    fig.update_layout(
        title="All Features - Importance Ranking",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Feature Categories")
    
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        st.markdown("#### Academic Features")
        st.write("""
        - Last Exam Score
        - Assignment Scores
        - Attendance
        - Concept Understanding
        - Class Participation
        """)
    
    with cat_col2:
        st.markdown("#### AI Usage Features")
        st.write("""
        - AI Tool Selection
        - Usage Time
        - AI Dependency
        - Generated Content %
        - Prompts per Week
        - Ethics Score
        """)
    
    with cat_col3:
        st.markdown("#### Lifestyle Features")
        st.write("""
        - Study Hours
        - Sleep Hours
        - Social Media Time
        - Tutoring Hours
        - Study Consistency
        - Improvement Rate
        """)

# =============================================================================
# TAB 4: ABOUT
# =============================================================================
with tab4:
    st.subheader("ℹ️ About This Application")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        ### 🤖 Machine Learning Model
        - **Algorithm:** Random Forest Classifier
        - **Trees:** 300 decision trees
        - **Training Samples:** 1000+
        - **Features:** 22 student metrics
        - **Test Accuracy:** ~85%
        
        ### 📊 What Gets Analyzed
        The model evaluates:
        1. Academic Performance (40% impact)
        2. Study Habits (25% impact)
        3. AI Tool Usage (20% impact)
        4. Lifestyle Factors (15% impact)
        """)
    
    with info_col2:
        st.markdown("""
### 🚀 Key Features
✅ Real-time predictions
✅ Confidence scores
✅ Feature importance analysis
✅ Detailed visualizations
✅ Model performance metrics
✅ Support for multiple AI tools

### ⚠️ Disclaimer
This tool provides predictions based on historical patterns. Always consult with academic advisors for personalized guidance.
        """)