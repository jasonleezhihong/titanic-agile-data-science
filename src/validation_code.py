# src/validation_code.py
import pandas as pd
import os
import sys

def validate_and_clean_data():
    """
    Validate data quality and remove duplicates
    """
    print("=" * 70)
    print("🔍 DATA VALIDATION AND CLEANING SCRIPT")
    print("=" * 70)
    
    # Input and output files
    input_file = 'data/titanic_train.csv'
    output_file = 'data/processed_dataset.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        print("Creating sample dataset for testing...")
        create_sample_dataset(input_file)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # ----- DATA VALIDATION CHECKS -----
    print("\n" + "-" * 70)
    print("📊 DATA VALIDATION CHECKS")
    print("-" * 70)
    
    validation_results = []
    
    # Check 1: Missing values
    print("\n1️⃣ CHECKING MISSING VALUES:")
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"⚠️ Missing values found in: {missing_cols}")
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            status = "FAIL" if missing_pct > 5 else "PASS"
            print(f"   {col}: {missing_count} ({missing_pct:.1f}%) - {status}")
            validation_results.append({
                'check': 'Missing Values',
                'column': col,
                'status': status,
                'details': f'{missing_count} ({missing_pct:.1f}%)'
            })
    else:
        print("✅ No missing values found!")
    
    # Check 2: Duplicate rows
    print("\n2️⃣ CHECKING DUPLICATE ROWS:")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"⚠️ Found {duplicate_count} duplicate rows")
        validation_results.append({
            'check': 'Duplicate Rows',
            'column': 'All',
            'status': 'FAIL',
            'details': f'{duplicate_count} duplicates found'
        })
    else:
        print("✅ No duplicate rows found!")
        validation_results.append({
            'check': 'Duplicate Rows',
            'column': 'All',
            'status': 'PASS',
            'details': 'No duplicates'
        })
    
    # Check 3: Data types
    print("\n3️⃣ CHECKING DATA TYPES:")
    expected_dtypes = {
        'Survived': 'int64',
        'Pclass': 'int64',
        'Age': 'float64',
        'SibSp': 'int64',
        'Parch': 'int64',
        'Fare': 'float64'
    }
    for col, expected in expected_dtypes.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            status = "PASS" if actual == expected else "FAIL"
            print(f"   {col}: {actual} (expected {expected}) - {status}")
            validation_results.append({
                'check': 'Data Type',
                'column': col,
                'status': status,
                'details': f'Expected {expected}, got {actual}'
            })
    
    # Check 4: Value ranges
    print("\n4️⃣ CHECKING VALUE RANGES:")
    if 'Age' in df.columns:
        invalid_age = df[(df['Age'] < 0) | (df['Age'] > 120)].shape[0]
        status = "PASS" if invalid_age == 0 else "FAIL"
        print(f"   Age: {invalid_age} invalid values - {status}")
        validation_results.append({
            'check': 'Value Range',
            'column': 'Age',
            'status': status,
            'details': f'{invalid_age} invalid values'
        })
    
    if 'Fare' in df.columns:
        invalid_fare = df[df['Fare'] < 0].shape[0]
        status = "PASS" if invalid_fare == 0 else "FAIL"
        print(f"   Fare: {invalid_fare} invalid values - {status}")
        validation_results.append({
            'check': 'Value Range',
            'column': 'Fare',
            'status': status,
            'details': f'{invalid_fare} invalid values'
        })
    
    # ----- REMOVE DUPLICATES -----
    print("\n" + "-" * 70)
    print("🔄 REMOVING DUPLICATES")
    print("-" * 70)
    
    original_count = len(df)
    if duplicate_count > 0:
        print(f"⚠️ Removing {duplicate_count} duplicate rows...")
        df_clean = df.drop_duplicates(keep='first')
        print(f"✅ Removed {duplicate_count} duplicates")
    else:
        print("✅ No duplicates to remove")
        df_clean = df.copy()
    
    # ----- SAVE CLEANED DATA -----
    print("\n" + "-" * 70)
    print("💾 SAVING CLEANED DATA")
    print("-" * 70)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)
    print(f"✅ Cleaned dataset saved to: {output_file}")
    print(f"   Rows: {len(df_clean)}")
    print(f"   Columns: {len(df_clean.columns)}")
    
    # ----- SUMMARY -----
    print("\n" + "=" * 70)
    print("📊 VALIDATION AND CLEANING SUMMARY")
    print("=" * 70)
    
    # Count passes and failures
    passed = sum(1 for r in validation_results if r['status'] == 'PASS')
    failed = sum(1 for r in validation_results if r['status'] == 'FAIL')
    
    print(f"\n✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"\n📈 Original rows: {original_count}")
    print(f"🗑️  Duplicates removed: {duplicate_count}")
    print(f"📈 Final rows: {len(df_clean)}")
    print(f"💾 Output file: {output_file}")
    
    # Show failed checks if any
    if failed > 0:
        print("\n⚠️ FAILED CHECKS:")
        for r in validation_results:
            if r['status'] == 'FAIL':
                print(f"   - {r['check']} ({r['column']}): {r['details']}")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    
    return df_clean

def create_sample_dataset(filename):
    """
    Create a sample dataset with duplicates for testing
    """
    import numpy as np
    
    print("Creating sample dataset...")
    np.random.seed(42)
    
    # Create sample data
    data = {
        'PassengerId': list(range(1, 101)),
        'Survived': np.random.choice([0, 1], 100),
        'Pclass': np.random.choice([1, 2, 3], 100),
        'Name': [f'Passenger_{i}' for i in range(1, 101)],
        'Sex': np.random.choice(['male', 'female'], 100),
        'Age': np.random.randint(1, 80, 100),
        'SibSp': np.random.randint(0, 5, 100),
        'Parch': np.random.randint(0, 3, 100),
        'Fare': np.random.uniform(10, 500, 100),
        'Embarked': np.random.choice(['S', 'C', 'Q'], 100)
    }
    df = pd.DataFrame(data)
    
    # Add some duplicates (first 5 rows)
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    # Save
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Sample dataset created: {len(df)} rows (including 5 duplicates)")

if __name__ == "__main__":
    try:
        validate_and_clean_data()
        print("\n✅ Script completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)