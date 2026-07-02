# src/validation_code.py
import pandas as pd
import os

def remove_duplicates(input_file, output_file):
    """
    Remove duplicate rows from dataset
    
    Parameters:
    - input_file: Path to input CSV file
    - output_file: Path to output CSV file (without duplicates)
    """
    print("=" * 60)
    print("🔍 DUPLICATE REMOVAL SCRIPT")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return False
    
    # Load dataset
    print(f"📂 Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\n📊 Duplicate rows found: {duplicate_count}")
    
    if duplicate_count > 0:
        print(f"⚠️ Removing {duplicate_count} duplicate rows...")
        # Remove duplicates (keep first occurrence)
        df_clean = df.drop_duplicates(keep='first')
        print(f"✅ After removal: {df_clean.shape[0]} rows")
    else:
        print("✅ No duplicates found! Dataset is clean.")
        df_clean = df.copy()
    
    # Save cleaned dataset
    print(f"\n💾 Saving cleaned dataset to: {output_file}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_clean.to_csv(output_file, index=False)
    print(f"✅ Dataset saved successfully!")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DUPLICATE REMOVAL SUMMARY")
    print("=" * 60)
    print(f"Original rows: {df.shape[0]}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Final rows: {df_clean.shape[0]}")
    print(f"Removal percentage: {(duplicate_count/df.shape[0]*100):.2f}%")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # Define input and output files
    input_file = "data/titanic_train.csv"  # Your original dataset
    output_file = "data/processed_dataset.csv"  # Cleaned dataset
    
    print("\n🚀 Starting duplicate removal process...\n")
    success = remove_duplicates(input_file, output_file)
    
    if success:
        print("\n✅ PROCESS COMPLETED SUCCESSFULLY!")
        exit(0)
    else:
        print("\n❌ PROCESS FAILED!")
        exit(1)