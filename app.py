# ========================
# Question 1
# ========================

# ------------------------
# Part (a)
# ------------------------
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('./data/titanic.csv')
print(df.head())

# dataset summary
print(df.info())
print(df.describe())

# ------------------------
# Part (b)
# ------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Distribution Analysis - Age distribution with survival
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of Age
axes[0].hist(df['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title('Age Distribution of Passengers', fontsize=14)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

# Boxplot of Age by Survival
sns.boxplot(x='Survived', y='Age', data=df, ax=axes[1])
axes[1].set_title('Age Distribution by Survival Status', fontsize=14)
axes[1].set_xticklabels(['Not Survived', 'Survived'])
axes[1].set_xlabel('Survival Status')
axes[1].set_ylabel('Age')

plt.tight_layout()
plt.savefig('./output/Question 2 - Age Distribution.png', dpi=300, bbox_inches='tight')
# plt.show()

# 2. Relationship Analysis - Correlation heatmap
plt.figure(figsize=(12, 8))

# Select numerical columns for correlation
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
correlation_matrix = df[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Numerical Variables', fontsize=16)
plt.savefig('./output/Question 2 - Heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()

# Key observation: Pclass (-0.34) and Fare (0.26) have strongest correlations with Survival.
# Age shows weak negative correlation (-0.08) with survival.

# 3. Categorical Analysis - Survival by Passenger Class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart - Survival by Pclass
survival_by_class = df.groupby('Pclass')['Survived'].agg(['count', 'mean'])
survival_by_class['count'].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Number of Passengers by Class', fontsize=14)
axes[0].set_xlabel('Passenger Class')
axes[0].set_ylabel('Count')

# Survival rate by class
survival_pct = df.groupby('Pclass')['Survived'].mean() * 100
survival_pct.plot(kind='bar', ax=axes[1], color=['red', 'orange', 'green'], edgecolor='black')
axes[1].set_title('Survival Rate by Passenger Class', fontsize=14)
axes[1].set_xlabel('Passenger Class')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].axhline(y=50, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('./output/Question 3 - Histogram.png', dpi=300, bbox_inches='tight')
# plt.show()


# ------------------------
# Part (c)
# ------------------------

# Check for missing value in each column
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

# Check for duplicated rows
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Check for invalid values for categorical columns
print("="*30)
print(df.dtypes)
print("="*30)
print("Unique values in 'Survived':", df['Survived'].unique())
print("Unique values in 'Sex':", df['Sex'].unique())
print("Unique values in 'Embarked':", df['Embarked'].unique())
print("Unique values in 'Pclass':", df['Pclass'].unique())
print("\n")

# Identify for outliers
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected")



# ========================
# Question 2
# ========================

# ------------------------
# Part (a)
# ------------------------

# Step 1: Create Family Size feature
print("===== FEATURE ENGINEERING STEP 1: FAMILY SIZE =====")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for self
print("FamilySize distribution:")
print(df['FamilySize'].value_counts().sort_index())
print("\nSurvival rate by Family Size:")
print(df.groupby('FamilySize')['Survived'].mean().sort_index())

# Step 2: Extract Title from Name
print("\n===== FEATURE ENGINEERING STEP 2: TITLE EXTRACTION =====")
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
print("Titles found in dataset:")
print(df['Title'].value_counts())

# Group rare titles
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
print("\nTitles after grouping rare ones:")
print(df['Title'].value_counts())

print("\nSurvival rate by Title:")
print(df.groupby('Title')['Survived'].mean().sort_values(ascending=False))


# ------------------------
# Part (b)
# ------------------------

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features].copy()
y = df['Survived']

# Handle missing values
X['Age'] = X['Age'].fillna(X['Age'].median())

# Encode Sex
encoder = LabelEncoder()
X['Sex'] = encoder.fit_transform(X['Sex'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train baseline model
baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_baseline = baseline_model.predict(X_test_scaled)

# Evaluate baseline model
print("Baseline Model Evaluation (Logistic Regression):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_baseline):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_baseline):.4f}")

# Convert scaled training data back to DataFrame
X_train_sm = pd.DataFrame(X_train_scaled, columns=features)

# Add intercept
X_train_sm = sm.add_constant(X_train_sm)

# Fit logistic regression
logit_model = sm.Logit(y_train.reset_index(drop=True), X_train_sm)
result = logit_model.fit()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': ['Intercept'] + features,
    'Coefficient': result.params.values,
    'P-value': result.pvalues.values
})

feature_importance['Decision (5%)'] = np.where(feature_importance['P-value'] < 0.05, "Significant","Not Significant")

feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)
print("\nFeature Importance (Coefficients):")
print(feature_importance)


# ------------------------
# Part (c)
# ------------------------

# Select features
features_improved = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Title']
X_improved = df[features_improved].copy()
y_improved = df['Survived']

# Handle missing values
X_improved['Age'] = X_improved['Age'].fillna(X_improved['Age'].median())

# Encode Sex
encoder = LabelEncoder()
X_improved['Sex'] = encoder.fit_transform(X_improved['Sex'])
X_improved['Title'] = encoder.fit_transform(X_improved['Title'])

# Split data
X_improved_train, X_improved_test, y_improved_train, y_improved_test = train_test_split(X_improved, y_improved, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_improved_train_scaled = scaler.fit_transform(X_improved_train)
X_improved_test_scaled = scaler.transform(X_improved_test)

# Train Improved model
improved_model = LogisticRegression(random_state=42)
improved_model.fit(X_improved_train_scaled, y_improved_train)

# Make predictions
y_pred_improved = improved_model.predict(X_improved_test_scaled)

# Evaluate Improved model
print("Improved Model Evaluation (Logistic Regression):")
print(f"Accuracy: {accuracy_score(y_improved_test, y_pred_improved):.4f}")
print(f"Precision: {precision_score(y_improved_test, y_pred_improved):.4f}")
print(f"Recall: {recall_score(y_improved_test, y_pred_improved):.4f}")
print(f"F1 Score: {f1_score(y_improved_test, y_pred_improved):.4f}")

# Convert scaled training data back to DataFrame
X_improved_train_sm = pd.DataFrame(X_improved_train_scaled, columns=features_improved)

# Add intercept
X_improved_train_sm = sm.add_constant(X_improved_train_sm)

# Fit logistic regression
logit_model_sm = sm.Logit(y_improved_train.reset_index(drop=True), X_improved_train_sm)
result_sm = logit_model_sm.fit()

# Feature importance
feature_importance_improved = pd.DataFrame({
    'Feature': ['Intercept'] + features_improved,
    'Coefficient': result_sm.params.values,
    'P-value': result_sm.pvalues.values
})

feature_importance_improved['Decision (5%)'] = np.where(feature_importance_improved['P-value'] < 0.05, "Significant","Not Significant")

feature_importance_improved = feature_importance_improved.sort_values('Coefficient', key=abs, ascending=False)
print("\nFeature Importance Improved (Coefficients):")
print(feature_importance_improved)


# ========================
# Question 4
# ========================

# ------------------------
# Part (a)
# ------------------------
# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Page configuration
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# Title
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("---")

# Load data
df = df.copy()

# Sidebar - Interactive features
st.sidebar.header("🔍 Interactive Filters")
st.sidebar.markdown("---")

# Filter 1: Passenger Class
pclass_filter = st.sidebar.multiselect(
    "Select Passenger Class(es):",
    options=sorted(df['Pclass'].unique()),
    default=sorted(df['Pclass'].unique())
)

# Filter 2: Gender
gender_filter = st.sidebar.multiselect(
    "Select Gender(s):",
    options=sorted(df['Sex'].unique()),
    default=sorted(df['Sex'].unique())
)

# Filter 3: Age Range
age_range = st.sidebar.slider(
    "Select Age Range:",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(0, 80)
)

# Filter data
filtered_df = df[
    (df['Pclass'].isin(pclass_filter)) &
    (df['Sex'].isin(gender_filter)) &
    (df['Age'].between(age_range[0], age_range[1]))
]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🔮 Predict Survival", "📋 Data"])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", len(filtered_df))
    col2.metric("Survival Rate", f"{filtered_df['Survived'].mean()*100:.1f}%")
    col3.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")
    col4.metric("Average Fare", f"${filtered_df['Fare'].mean():.2f}")
    
    # Data summary
    st.subheader("Data Summary")
    st.dataframe(filtered_df.describe())

with tab2:
    st.header("Visualizations")
    
    # Visualization 1: Survival by Class
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Survival by Passenger Class")
        fig, ax = plt.subplots(figsize=(8, 5))
        survival_by_class = filtered_df.groupby('Pclass')['Survived'].mean() * 100
        survival_by_class.plot(kind='bar', color=['#ff6b6b', '#ffd93d', '#6bcb77'], ax=ax)
        ax.set_ylabel("Survival Rate (%)")
        ax.set_xlabel("Passenger Class")
        ax.set_ylim(0, 100)
        for i, v in enumerate(survival_by_class):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Survival by Gender")
        fig, ax = plt.subplots(figsize=(8, 5))
        survival_by_gender = filtered_df.groupby('Sex')['Survived'].mean() * 100
        survival_by_gender.plot(kind='bar', color=['#4ecdc4', '#ffe66d'], ax=ax)
        ax.set_ylabel("Survival Rate (%)")
        ax.set_xlabel("Gender")
        ax.set_ylim(0, 100)
        for i, v in enumerate(survival_by_gender):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center')
        st.pyplot(fig)
    
    # Visualization 3: Age Distribution
    st.subheader("Age Distribution by Survival")
    fig, ax = plt.subplots(figsize=(10, 6))
    for survived in [0, 1]:
        subset = filtered_df[filtered_df['Survived'] == survived]['Age'].dropna()
        label = "Did Not Survive" if survived == 0 else "Survived"
        ax.hist(subset, bins=30, alpha=0.6, label=label)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.header("🔮 Predict Survival")
    
    st.markdown("### Enter Passenger Details")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    
    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=8, value=0)
        parch = st.number_input("Number of Parents/Children", min_value=0, max_value=6, value=0)
        fare = st.number_input("Fare Amount", min_value=0.0, max_value=500.0, value=32.0)
    
    # Model prediction
    if st.button("Predict Survival", type="primary"):
        # Prepare features
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare]
        })
        
        # Load model (assuming you've saved it)
        # For demo, using a simple logistic regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        
        # Quick training for demo
        X_demo = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna())
        y_demo = df.loc[X_demo.index, 'Survived']
        X_demo['Age'] = X_demo['Age'].fillna(X_demo['Age'].median())
        model.fit(X_demo, y_demo)
        
        # Predict
        input_processed = pd.get_dummies(input_data)
        input_processed = input_processed.reindex(columns=X_demo.columns, fill_value=0)
        
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]
        
        # Display result
        st.markdown("---")
        if prediction == 1:
            st.success(f"✅ Predicted: **SURVIVED** (Probability: {probability:.2%})")
        else:
            st.error(f"❌ Predicted: **DID NOT SURVIVE** (Probability: {1-probability:.2%})")

with tab4:
    st.header("📋 Data Viewer")
    st.dataframe(filtered_df)