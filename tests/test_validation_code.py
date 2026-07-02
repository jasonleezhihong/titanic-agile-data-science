# tests/test_validation_code.py
import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.remove_duplicates import remove_duplicates

class TestValidationCode(unittest.TestCase):
    """Test cases for data validation functionality"""
    
    def setUp(self):
        """Create test data with duplicates"""
        # Create sample data with duplicates
        self.test_data = pd.DataFrame({
            'Name': ['John', 'Jane', 'Bob', 'John', 'Alice', 'Bob'],
            'Age': [25, 30, 35, 25, 28, 35],
            'City': ['NYC', 'LA', 'Chicago', 'NYC', 'SF', 'Chicago']
        })
        
        # Save test data
        os.makedirs('test_data', exist_ok=True)
        self.test_input = 'test_data/test_with_duplicates.csv'
        self.test_output = 'test_data/test_cleaned.csv'
        self.test_data.to_csv(self.test_input, index=False)
    
    def test_remove_duplicates(self):
        """Test that duplicates are correctly removed"""
        # Run duplicate removal
        result = remove_duplicates(self.test_input, self.test_output)
        
        # Check if function ran successfully
        self.assertTrue(result)
        
        # Check if output file exists
        self.assertTrue(os.path.exists(self.test_output))
        
        # Load cleaned data
        cleaned_df = pd.read_csv(self.test_output)
        
        # Check number of rows (should be 4 unique rows, not 6)
        self.assertEqual(len(cleaned_df), 4)
        
        # Check that no duplicates remain
        duplicate_count = cleaned_df.duplicated().sum()
        self.assertEqual(duplicate_count, 0)
        
        print("✅ Test passed: Duplicates removed successfully")
    
    def test_no_duplicates(self):
        """Test that function handles datasets with no duplicates correctly"""
        # Create dataset without duplicates
        no_dup_data = pd.DataFrame({
            'Name': ['John', 'Jane', 'Bob', 'Alice'],
            'Age': [25, 30, 35, 28],
            'City': ['NYC', 'LA', 'Chicago', 'SF']
        })
        no_dup_input = 'test_data/test_no_duplicates.csv'
        no_dup_output = 'test_data/test_no_dup_cleaned.csv'
        no_dup_data.to_csv(no_dup_input, index=False)
        
        # Run duplicate removal
        result = remove_duplicates(no_dup_input, no_dup_output)
        self.assertTrue(result)
        
        # Check that no rows were removed (still 4 rows)
        cleaned_df = pd.read_csv(no_dup_output)
        self.assertEqual(len(cleaned_df), 4)
        self.assertEqual(cleaned_df.duplicated().sum(), 0)
        
        print("✅ Test passed: Handled no-duplicate dataset correctly")
    
    def test_invalid_file(self):
        """Test that function handles missing files correctly"""
        # Try to process a non-existent file
        result = remove_duplicates('non_existent.csv', 'test_data/output.csv')
        self.assertFalse(result)
        
        print("✅ Test passed: Handles missing files correctly")
    
    def tearDown(self):
        """Clean up test files"""
        # Remove test files
        import shutil
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')

if __name__ == "__main__":
    unittest.main()