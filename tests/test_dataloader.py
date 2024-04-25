import unittest
import pandas as pd
from pathlib import Path
import os
import shutil

# Assuming your function is in a file named dataloader.py
from dataloader import create_csv_data

class TestCreateCSVData(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path('test_dir')
        self.test_dir.mkdir(exist_ok=True)

        # Create some dummy image and label files
        self.image_files = [self.test_dir / f'test{i}.jpg' for i in range(5)]
        self.label_files = [self.test_dir / f'test{i}.txt' for i in range(5)]

        for file in self.image_files + self.label_files:
            file.touch()

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_create_csv_data(self):
        # Call the function with the test directory
        create_csv_data(self.test_dir, self.test_dir)

        # Check that a CSV file was created
        self.assertTrue((self.test_dir / 'data.csv').exists())

        # Load the CSV file into a DataFrame
        df = pd.read_csv(self.test_dir / 'data.csv')

        # Check that the DataFrame has the correct columns
        self.assertEqual(set(df.columns), {'image', 'label', 'bbox'})

        # Check that the DataFrame has the correct number of rows
        self.assertEqual(len(df), len(self.image_files))

        # Check that the image paths in the DataFrame are correct
        self.assertEqual(set(df['image']), {str(file) for file in self.image_files})

        # Check that the labels in the DataFrame are correct
        self.assertEqual(set(df['label']), {file.parent.name for file in self.image_files})

        # Check that the bbox in the DataFrame are correct
        self.assertEqual(set(df['bbox']), {str(file) for file in self.label_files})


def get_item_dataloader(dataloader):
    return next(iter(dataloader))

if __name__ == '__main__':
    unittest.main()