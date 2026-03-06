import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DS Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('./data/data_science_salaries.csv')
    return df

with st.spinner('📂 Loading data...'):
    df = load_data()

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def feature_engineering(df):
    df_fe = df.copy()

    exp_map          = {'Entry-level': 0, 'Mid-level': 1, 'Senior-level': 2, 'Executive-level': 3}
    employment_map   = {'Part-time': 0, 'Freelance': 1, 'Contract': 2, 'Full-time': 3}
    company_size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
    work_model_map   = {'On-site': 0, 'Hybrid': 50, 'Remote': 100}

    df_fe['experience_rank']   = df_fe['experience_level'].map(exp_map)
    df_fe['employment_rank']   = df_fe['employment_type'].map(employment_map)
    df_fe['company_size_rank'] = df_fe['company_size'].map(company_size_map)
    df_fe['remote_ratio']      = df_fe['work_models'].map(work_model_map)

    def categorize_job(title):
        t = title.lower()
        if any(x in t for x in ['machine learning', 'ml engineer', 'mlops']):
            return 'ML Engineering'
        elif any(x in t for x in ['data scientist', 'research scientist']):
            return 'Data Science'
        elif any(x in t for x in ['data engineer', 'etl', 'pipeline']):
            return 'Data Engineering'
        elif any(x in t for x in ['analyst', 'analytics']):
            return 'Analytics'
        elif any(x in t for x in ['manager', 'director', 'head', 'lead', 'vp', 'chief']):
            return 'Management'
        elif 'architect' in t:
            return 'Architecture'
        elif any(x in t for x in ['ai', 'nlp', 'computer vision']):
            return 'AI / Research'
        elif any(x in t for x in ['bi', 'business intelligence']):
            return 'BI / Reporting'
        else:
            return 'Other'

    df_fe['job_category'] = df_fe['job_title'].apply(categorize_job)

    residence_freq = df_fe['employee_residence'].value_counts() / len(df_fe)
    location_freq  = df_fe['company_location'].value_counts()  / len(df_fe)
    df_fe['residence_freq']        = df_fe['employee_residence'].map(residence_freq)
    df_fe['company_location_freq'] = df_fe['company_location'].map(location_freq)

    df_fe['exp_x_company_size'] = df_fe['experience_rank'] * df_fe['company_size_rank']
    df_fe['is_local']           = (df_fe['employee_residence'] == df_fe['company_location']).astype(int)
    df_fe['remote_large']       = ((df_fe['work_models'] == 'Remote') & (df_fe['company_size'] == 'Large')).astype(int)
    df_fe['year_recency']       = df_fe['work_year'] - df_fe['work_year'].min()

    return df_fe, residence_freq, location_freq

with st.spinner('⚙️ Engineering features...'):
    df_fe, residence_freq, location_freq = feature_engineering(df)

# ─────────────────────────────────────────────
#  TARGET ENCODING + FEATURE MATRIX
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_features(df_fe):
    from category_encoders import TargetEncoder

    te = TargetEncoder(cols=['job_title', 'job_category'], smoothing=10)
    encoded = te.fit_transform(
        df_fe[['job_title', 'job_category']],
        df_fe['salary_in_usd']
    )
    df_fe['job_title_encoded']    = encoded['job_title']
    df_fe['job_category_encoded'] = encoded['job_category']

    feature_cols = [
        'experience_rank', 'employment_rank', 'company_size_rank',
        'remote_ratio', 'year_recency', 'residence_freq',
        'company_location_freq', 'exp_x_company_size', 'is_local',
        'remote_large', 'job_title_encoded', 'job_category_encoded',
    ]

    X = df_fe[feature_cols].copy()
    y = np.log1p(df_fe['salary_in_usd'])

    mask = X.isnull().any(axis=1)
    X    = X[~mask].reset_index(drop=True)
    y    = y[~mask].reset_index(drop=True)

    return X, y, te, feature_cols

with st.spinner('🔢 Building feature matrix...'):
    X, y, te, feature_cols = build_features(df_fe)

# ─────────────────────────────────────────────
#  TRAIN MODELS + WEIGHTED ENSEMBLE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    import xgboost as xgb
    import lightgbm as lgb

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1
    )
    lgb_model.fit(X_train, y_train)

    gb_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42
    )
    gb_model.fit(X_train, y_train)

    r2_xgb = r2_score(y_test, xgb_model.predict(X_test))
    r2_lgb = r2_score(y_test, lgb_model.predict(X_test))
    r2_gb  = r2_score(y_test, gb_model.predict(X_test))
    total  = r2_xgb + r2_lgb + r2_gb
    w_xgb  = r2_xgb / total
    w_lgb  = r2_lgb / total
    w_gb   = r2_gb  / total

    def ensemble_predict(X_input):
        return (
            w_xgb * xgb_model.predict(X_input) +
            w_lgb * lgb_model.predict(X_input) +
            w_gb  * gb_model.predict(X_input)
        )

    ensemble_preds = ensemble_predict(X_test)
    r2_ensemble    = r2_score(y_test, ensemble_preds)
    rmse_ensemble  = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    rmse_usd       = int(np.sqrt(np.mean(
        (np.expm1(ensemble_preds) - np.expm1(y_test)) ** 2
    )))

    metrics = {
        'r2_xgb': round(r2_xgb, 4), 'r2_lgb': round(r2_lgb, 4),
        'r2_gb':  round(r2_gb,  4), 'r2_ensemble': round(r2_ensemble, 4),
        'rmse': round(rmse_ensemble, 4), 'rmse_usd': rmse_usd,
        'w_xgb': round(w_xgb, 3), 'w_lgb': round(w_lgb, 3),
        'w_gb':  round(w_gb,  3),
    }

    return (xgb_model, lgb_model, gb_model,
            ensemble_predict, X_train, X_test,
            y_train, y_test, metrics)

with st.spinner('🤖 Training models...'):
    (xgb_model, lgb_model, gb_model,
     ensemble_predict, X_train, X_test,
     y_train, y_test, metrics) = train_model(X, y)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title('💼 DS Salary Predictor')
st.markdown('Predict your Data Science salary based on your profile.')
st.markdown('---')

# ─────────────────────────────────────────────
#  SIDEBAR INPUTS  ← must come before tabs
# ─────────────────────────────────────────────
st.sidebar.title('🧑‍💻 Your Profile')

job_title = st.sidebar.selectbox(
    'Job Title',
    sorted(df['job_title'].unique().tolist()),
    index=0
)
experience_level = st.sidebar.selectbox(
    'Experience Level',
    ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level'],
    index=2
)
employment_type = st.sidebar.selectbox(
    'Employment Type',
    ['Full-time', 'Part-time', 'Contract', 'Freelance'],
    index=0
)
company_size = st.sidebar.selectbox(
    'Company Size',
    ['Small', 'Medium', 'Large'],
    index=1
)
work_model = st.sidebar.selectbox(
    'Work Model',
    ['On-site', 'Hybrid', 'Remote'],
    index=2
)
work_year = st.sidebar.slider(
    'Work Year',
    min_value=int(df['work_year'].min()),
    max_value=int(df['work_year'].max()),
    value=int(df['work_year'].max())
)
employee_residence = st.sidebar.selectbox(
    'Employee Residence',
    sorted(df['employee_residence'].unique().tolist()),
    index=0
)
company_location = st.sidebar.selectbox(
    'Company Location',
    sorted(df['company_location'].unique().tolist()),
    index=0
)

predict_btn = st.sidebar.button('🔮 Predict My Salary', use_container_width=True)

# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
def build_input_row(job_title, experience_level, employment_type,
                    company_size, work_model, work_year,
                    employee_residence, company_location):

    exp_map          = {'Entry-level': 0, 'Mid-level': 1, 'Senior-level': 2, 'Executive-level': 3}
    employment_map   = {'Part-time': 0, 'Freelance': 1, 'Contract': 2, 'Full-time': 3}
    company_size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
    work_model_map   = {'On-site': 0, 'Hybrid': 50, 'Remote': 100}

    exp_rank   = exp_map[experience_level]
    emp_rank   = employment_map[employment_type]
    size_rank  = company_size_map[company_size]
    remote_val = work_model_map[work_model]
    year_rec   = work_year - int(df['work_year'].min())
    res_freq   = residence_freq.get(employee_residence, residence_freq.mean())
    loc_freq   = location_freq.get(company_location,   location_freq.mean())

    job_cat    = df_fe[df_fe['job_title'] == job_title]['job_category'].values
    job_cat    = job_cat[0] if len(job_cat) > 0 else 'Other'
    te_input   = pd.DataFrame({'job_title': [job_title], 'job_category': [job_cat]})
    te_out     = te.transform(te_input)

    row = {
        'experience_rank':       exp_rank,
        'employment_rank':       emp_rank,
        'company_size_rank':     size_rank,
        'remote_ratio':          remote_val,
        'year_recency':          year_rec,
        'residence_freq':        res_freq,
        'company_location_freq': loc_freq,
        'exp_x_company_size':    exp_rank * size_rank,
        'is_local':              int(employee_residence == company_location),
        'remote_large':          int(work_model == 'Remote' and company_size == 'Large'),
        'job_title_encoded':     te_out['job_title'].values[0],
        'job_category_encoded':  te_out['job_category'].values[0],
    }

    return pd.DataFrame([row])[feature_cols]

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_predict, tab_eda, tab_model, tab_shap = st.tabs([
    '🔮 Predict', '📊 EDA', '🤖 Model Performance', '🔬 SHAP'
])

# ══════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:
    if not predict_btn:
        st.info('👈 Fill in your profile in the sidebar and click **Predict My Salary**')
    else:
        input_df  = build_input_row(
            job_title, experience_level, employment_type,
            company_size, work_model, work_year,
            employee_residence, company_location
        )

        log_pred  = ensemble_predict(input_df)[0]
        pred_usd  = np.expm1(log_pred)
        pred_low  = np.expm1(log_pred - metrics['rmse'])
        pred_high = np.expm1(log_pred + metrics['rmse'])

        st.markdown('### 💰 Predicted Salary')
        col1, col2, col3 = st.columns(3)
        col1.metric('Estimated Salary', f'${pred_usd:,.0f}')
        col2.metric('Lower Bound',      f'${pred_low:,.0f}')
        col3.metric('Upper Bound',      f'${pred_high:,.0f}')
        st.markdown('---')

        st.markdown('### 📋 Your Profile')
        profile = {
            'Field': ['Job Title', 'Experience', 'Employment',
                      'Company Size', 'Work Model', 'Year',
                      'Residence', 'Company Location'],
            'Value': [job_title, experience_level, employment_type,
                      company_size, work_model, work_year,
                      employee_residence, company_location]
        }
        st.dataframe(pd.DataFrame(profile), hide_index=True, use_container_width=True)
        st.markdown('---')

        st.markdown('### 📍 How You Compare to Market')
        similar = df[df['experience_level'] == experience_level]['salary_in_usd']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Your Prediction',  f'${pred_usd:,.0f}')
        c2.metric('Market Median',    f'${similar.median():,.0f}')
        c3.metric('Market 25th %ile', f'${similar.quantile(0.25):,.0f}')
        c4.metric('Market 75th %ile', f'${similar.quantile(0.75):,.0f}')

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(similar, bins=30, color='#4f8ef7', edgecolor='white', alpha=0.7)
        ax.axvline(pred_usd, color='#e94560', lw=2.5, linestyle='--',
                   label=f'Your Prediction: ${pred_usd:,.0f}')
        ax.axvline(similar.median(), color='#10b981', lw=1.5, linestyle=':',
                   label=f'Market Median: ${similar.median():,.0f}')
        ax.set_xlabel('Salary (USD)')
        ax.set_title(f'Your Salary vs Market — {experience_level}')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
#  TAB 2 — EDA
# ══════════════════════════════════════════════
with tab_eda:
    st.markdown('### 📊 Exploratory Data Analysis')

    # ── Overview metrics ──────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Records',    f'{len(df):,}')
    c2.metric('Unique Job Titles', df['job_title'].nunique())
    c3.metric('Countries',         df['company_location'].nunique())
    c4.metric('Years Covered',     f"{df['work_year'].min()}–{df['work_year'].max()}")

    st.markdown('---')

    # ── Salary by Experience ──────────────────
    st.markdown('#### 💰 Salary by Experience Level')
    exp_order = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    df.groupby('experience_level')['salary_in_usd'].median().reindex(exp_order).plot(
        kind='bar', ax=ax1,
        color=['#4f8ef7','#10b981','#f59e0b','#e94560'],
        edgecolor='white'
    )
    ax1.set_title('Median Salary by Experience Level', fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Median Salary (USD)')
    ax1.tick_params(rotation=15)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown('---')

    # ── Salary by Work Model ──────────────────
    st.markdown('#### 🏠 Salary by Work Model')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df.groupby('work_models')['salary_in_usd'].median().sort_values().plot(
        kind='bar', ax=ax2,
        color=['#4f8ef7','#10b981','#e94560'],
        edgecolor='white'
    )
    ax2.set_title('Median Salary by Work Model', fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Median Salary (USD)')
    ax2.tick_params(rotation=0)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown('---')

    # ── Salary by Company Size ────────────────
    st.markdown('#### 🏢 Salary by Company Size')
    size_order = ['Small', 'Medium', 'Large']
    fig3, ax3  = plt.subplots(figsize=(8, 4))
    df.groupby('company_size')['salary_in_usd'].median().reindex(size_order).plot(
        kind='bar', ax=ax3,
        color=['#4f8ef7','#10b981','#e94560'],
        edgecolor='white'
    )
    ax3.set_title('Median Salary by Company Size', fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_ylabel('Median Salary (USD)')
    ax3.tick_params(rotation=0)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown('---')

    # ── Salary Trend by Year ──────────────────
    st.markdown('#### 📈 Salary Trend by Year')
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    df.groupby('work_year')['salary_in_usd'].median().plot(
        kind='line', ax=ax4,
        marker='o', color='#4f8ef7', linewidth=2.5
    )
    ax4.set_title('Median Salary Trend by Year', fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Median Salary (USD)')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown('---')

    # ── Top 10 Job Titles ─────────────────────
    st.markdown('#### 🏆 Top 10 Job Titles by Median Salary')
    top_titles = (
        df.groupby('job_title')['salary_in_usd']
        .median()
        .sort_values(ascending=False)
        .head(10)
    )
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    top_titles.plot(kind='barh', ax=ax5, color='#4f8ef7', edgecolor='white')
    ax5.invert_yaxis()
    ax5.set_title('Top 10 Job Titles by Median Salary', fontweight='bold')
    ax5.set_xlabel('Median Salary (USD)')
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.markdown('---')

    # ── Interactive filter ────────────────────
    st.markdown('#### 🔍 Filter & Explore')
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        sel_exp = st.multiselect(
            'Experience Level', exp_order, default=exp_order)
    with col_f2:
        sel_wm = st.multiselect(
            'Work Model',
            df['work_models'].unique().tolist(),
            default=df['work_models'].unique().tolist()
        )
    with col_f3:
        sel_size = st.multiselect(
            'Company Size', size_order, default=size_order)

    filtered = df[
        (df['experience_level'].isin(sel_exp)) &
        (df['work_models'].isin(sel_wm)) &
        (df['company_size'].isin(sel_size))
    ]

    st.caption(f'{len(filtered):,} records match your filters')

    if len(filtered) > 0:
        fig6, ax6 = plt.subplots(figsize=(10, 4))
        ax6.hist(filtered['salary_in_usd'], bins=30,
                 color='#4f8ef7', edgecolor='white', alpha=0.8)
        ax6.axvline(filtered['salary_in_usd'].median(),
                    color='#e94560', lw=2, linestyle='--',
                    label=f"Median: ${filtered['salary_in_usd'].median():,.0f}")
        ax6.set_xlabel('Salary (USD)')
        ax6.set_title('Filtered Salary Distribution', fontweight='bold')
        ax6.legend()
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()
    else:
        st.warning('No records match your filters.')


# ══════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab_model:
    st.markdown('### 🤖 Model Performance')

    # ── Model metrics summary ─────────────────
    st.markdown('#### 📊 Model Comparison')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('XGBoost R²',       str(metrics['r2_xgb']))
    c2.metric('LightGBM R²',      str(metrics['r2_lgb']))
    c3.metric('Gradient Boost R²',str(metrics['r2_gb']))
    c4.metric('Ensemble R²',      str(metrics['r2_ensemble']))

    st.markdown('---')

    # ── Weights ───────────────────────────────
    st.markdown('#### ⚖️ Ensemble Weights (based on R²)')
    col1, col2, col3 = st.columns(3)
    col1.metric('XGBoost Weight',        str(metrics['w_xgb']))
    col2.metric('LightGBM Weight',       str(metrics['w_lgb']))
    col3.metric('Gradient Boost Weight', str(metrics['w_gb']))

    st.markdown('---')

    # ── R² bar chart ──────────────────────────
    st.markdown('#### 📈 R² Score by Model')
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    model_names = ['XGBoost', 'LightGBM', 'Gradient Boost', 'Ensemble']
    r2_values   = [metrics['r2_xgb'], metrics['r2_lgb'],
                   metrics['r2_gb'],  metrics['r2_ensemble']]
    colors      = ['#4f8ef7', '#10b981', '#f59e0b', '#e94560']

    bars = ax1.barh(model_names, r2_values, color=colors, edgecolor='white')
    ax1.set_xlabel('R² Score')
    ax1.set_title('R² Score by Model (higher is better)', fontweight='bold')
    ax1.set_xlim(0, max(r2_values) * 1.1)
    for bar, val in zip(bars, r2_values):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown('---')

    # ── Actual vs Predicted ───────────────────
    st.markdown('#### 🎯 Actual vs Predicted (Ensemble)')
    ensemble_preds = ensemble_predict(X_test)

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scatter
    axes[0].scatter(y_test, ensemble_preds,
                    alpha=0.4, color='#4f8ef7', s=20)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual log(Salary)')
    axes[0].set_ylabel('Predicted log(Salary)')
    axes[0].set_title('Actual vs Predicted')

    # Residuals distribution
    residuals = y_test - ensemble_preds
    axes[1].hist(residuals, bins=40, color='#10b981', edgecolor='white')
    axes[1].axvline(0, color='#e94560', lw=2, linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_title('Residual Distribution')

    # Residuals vs Fitted
    axes[2].scatter(ensemble_preds, residuals,
                    alpha=0.4, color='#f59e0b', s=20)
    axes[2].axhline(0, color='#e94560', lw=2, linestyle='--')
    axes[2].set_xlabel('Fitted Values')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title('Residuals vs Fitted')

    plt.suptitle('Ensemble — Residual Analysis', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown('---')

    # ── Feature importance ────────────────────
    st.markdown('#### 🔑 Feature Importance (Gradient Boosting)')
    importance_df = pd.DataFrame({
        'Feature':    feature_cols,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.barh(importance_df['Feature'], importance_df['Importance'],
             color='#4f8ef7', edgecolor='white')
    ax3.set_title('Feature Importance', fontweight='bold')
    ax3.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown('---')

    # ── Known limitations ─────────────────────
    st.markdown('#### ⚠️ Known Limitations')
    st.warning("""
    - **R² ~0.51** — model explains 51% of salary variance. 
      Salary data has high natural variation across countries and roles.
    - **Executive-level** predictions are less reliable due to 
      low sample count (n=254 total).
    - Model does not account for skills, education, or city-level data.
    """)

# ══════════════════════════════════════════════
#  TAB 4 — SHAP
# ══════════════════════════════════════════════
with tab_shap:
    st.markdown('### 🔬 SHAP — Model Explainability')
    st.markdown("""
    SHAP explains **why** the model made each prediction.  
    It shows how much each feature pushed the salary **up** or **down**
    from the average.
    """)

    st.markdown('---')

    import shap

    # ── Global SHAP — Feature Importance ─────
    st.markdown('#### 📊 Global Feature Importance')

    with st.spinner('Computing SHAP values...'):
        explainer  = shap.TreeExplainer(gb_model)
        shap_vals  = explainer.shap_values(X_test)

    # Bar chart
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    order_idx  = np.argsort(mean_shap)
    ax1.barh(
        [feature_cols[i] for i in order_idx],
        mean_shap[order_idx],
        color='#4f8ef7', edgecolor='white'
    )
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title('Global Feature Importance (SHAP)', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown('---')

    # ── Beeswarm ──────────────────────────────
    st.markdown('#### 🐝 Beeswarm — Feature Impact Direction')
    st.caption('Red = high feature value | Blue = low feature value | Right = increases salary')

    shap_explanation = shap.Explanation(
        values=shap_vals,
        base_values=np.full(len(shap_vals), explainer.expected_value),
        data=X_test.values,
        feature_names=feature_cols
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_explanation, show=False)
    plt.title('SHAP Beeswarm', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown('---')

    # ── Single prediction SHAP ────────────────
    st.markdown('#### 🔍 Explain a Single Prediction')
    st.caption('Select a row from the test set to see why that salary was predicted.')

    row_idx = st.slider('Test set row index', 0, len(X_test) - 1, 0)

    shap_exp_single = shap.Explanation(
        values=shap_vals[row_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[row_idx],
        feature_names=feature_cols
    )

    pred_single = np.expm1(ensemble_predict(X_test.iloc[[row_idx]])[0])
    actual_single = np.expm1(y_test.iloc[row_idx])

    col1, col2 = st.columns(2)
    col1.metric('Predicted Salary', f'${pred_single:,.0f}')
    col2.metric('Actual Salary',    f'${actual_single:,.0f}')

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_exp_single, show=False)
    plt.title(f'SHAP Waterfall — Row {row_idx}', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown('---')

    # ── SHAP interpretation guide ─────────────
    st.markdown('#### 📖 How to Read SHAP')
    st.info("""
    - **Positive SHAP value** → feature pushed salary **higher** than average  
    - **Negative SHAP value** → feature pushed salary **lower** than average  
    - **Base value** → average predicted salary across all training data  
    - **f(x)** → final predicted salary for this specific row  
    """)