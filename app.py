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
# Question 3
# ========================

# ------------------------
# Part (a)
# ------------------------

def validate_dataset(df):
    validation_results = []
    
    # Check 1: Missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    for col in missing_cols:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        validation_results.append({
            'Check': 'Missing Values',
            'Column': col,
            'Status': 'FAIL' if missing_pct > 5 else 'PASS',
            'Details': f'{missing_pct:.1f}% missing'
        })
    
    # Check 2: Duplicate rows
    duplicate_count = df.duplicated().sum()
    validation_results.append({
        'Check': 'Duplicate Rows',
        'Column': 'All',
        'Status': 'FAIL' if duplicate_count > 0 else 'PASS',
        'Details': f'{duplicate_count} duplicate rows'
    })
    
    # Check 3: Data types
    expected_dtypes = {
        'Survived': 'int64',
        'Pclass': 'int64',
        'Name': 'object',
        'Sex': 'object',
        'Age': 'float64',
        'SibSp': 'int64',
        'Parch': 'int64',
        'Ticket': 'object',
        'Fare': 'float64',
        'Cabin': 'object',
        'Embarked': 'object'
    }
    
    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            validation_results.append({
                'Check': 'Data Type',
                'Column': col,
                'Status': 'PASS' if actual_dtype == expected_dtype else 'FAIL',
                'Details': f'Expected {expected_dtype}, got {actual_dtype}'
            })
    
    # Check 4: Value ranges
    # Age should be between 0 and 120
    if 'Age' in df.columns:
        invalid_age = df[(df['Age'] < 0) | (df['Age'] > 120)].shape[0]
        validation_results.append({
            'Check': 'Value Range',
            'Column': 'Age',
            'Status': 'FAIL' if invalid_age > 0 else 'PASS',
            'Details': f'{invalid_age} invalid age values'
        })
    
    # Fare should be non-negative
    if 'Fare' in df.columns:
        invalid_fare = df[df['Fare'] < 0].shape[0]
        validation_results.append({
            'Check': 'Value Range',
            'Column': 'Fare',
            'Status': 'FAIL' if invalid_fare > 0 else 'PASS',
            'Details': f'{invalid_fare} invalid fare values'
        })
    
    # Create summary DataFrame
    results_df = pd.DataFrame(validation_results)
    print(results_df.to_string(index=False))
    
    # Return overall status
    overall_status = 'PASS' if all(results_df['Status'] == 'PASS') else 'FAIL'
    print(f"\nOverall Validation Status: {overall_status}")
    
    return results_df

validate_dataset(df)