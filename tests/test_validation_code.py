import unittest
import pandas as pd
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation_code import validate_and_clean_data


class TestValidationCode(unittest.TestCase):

    def setUp(self):

        os.makedirs("test_data", exist_ok=True)

        self.test_input = "test_data/test_with_duplicates.csv"
        self.test_output = "test_data/test_cleaned.csv"

        df = pd.DataFrame({
            "Name": ["John", "Jane", "Bob", "John", "Alice", "Bob"],
            "Age": [25, 30, 35, 25, 28, 35],
            "City": ["NYC", "LA", "Chicago", "NYC", "SF", "Chicago"]
        })

        df.to_csv(self.test_input, index=False)

    def tearDown(self):

        if os.path.exists("test_data"):
            shutil.rmtree("test_data")

    def test_remove_duplicates(self):

        result = validate_and_clean_data(
            self.test_input,
            self.test_output
        )

        self.assertTrue(result)

        self.assertTrue(os.path.exists(self.test_output))

        cleaned = pd.read_csv(self.test_output)

        self.assertEqual(len(cleaned), 4)

        self.assertEqual(cleaned.duplicated().sum(), 0)

    def test_no_duplicates(self):

        input_file = "test_data/no_dup.csv"
        output_file = "test_data/no_dup_clean.csv"

        df = pd.DataFrame({
            "Name": ["John", "Jane", "Bob", "Alice"],
            "Age": [25, 30, 35, 28],
            "City": ["NYC", "LA", "Chicago", "SF"]
        })

        df.to_csv(input_file, index=False)

        result = validate_and_clean_data(
            input_file,
            output_file
        )

        self.assertTrue(result)

        cleaned = pd.read_csv(output_file)

        self.assertEqual(len(cleaned), 4)

        self.assertEqual(cleaned.duplicated().sum(), 0)

    def test_invalid_file(self):

        result = validate_and_clean_data(
            "does_not_exist.csv",
            "test_data/output.csv"
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()