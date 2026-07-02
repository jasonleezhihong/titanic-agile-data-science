import pandas as pd
import os
import sys


def validate_and_clean_data(
    input_file="data/titanic_train.csv",
    output_file="data/processed_dataset.csv"
):
    """
    Validate data quality, remove duplicates and save cleaned dataset.

    Returns:
        bool: True if successful, False otherwise.
    """

    print("=" * 70)
    print("DATA VALIDATION AND CLEANING")
    print("=" * 70)

    # Check file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return False

    try:
        # Load dataset
        df = pd.read_csv(input_file)

        print(f"Loaded {len(df)} rows.")

        # ----------------------------
        # Missing Values
        # ----------------------------
        print("\nChecking Missing Values")

        for col in df.columns:
            missing = df[col].isnull().sum()

            if missing > 0:
                pct = missing / len(df) * 100
                print(f"{col}: {missing} ({pct:.2f}%)")

        # ----------------------------
        # Duplicate Rows
        # ----------------------------
        duplicate_count = df.duplicated().sum()

        print(f"\nDuplicate rows: {duplicate_count}")

        if duplicate_count > 0:
            df = df.drop_duplicates()

        # ----------------------------
        # Create output directory
        # ----------------------------
        output_dir = os.path.dirname(output_file)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # ----------------------------
        # Save cleaned data
        # ----------------------------
        df.to_csv(output_file, index=False)

        print(f"Saved cleaned dataset to {output_file}")

        return True

    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":

    success = validate_and_clean_data()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)