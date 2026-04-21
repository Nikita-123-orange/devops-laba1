import configparser
import os
import unittest
from unittest.mock import patch
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from src.preprocess import DataMaker

config = configparser.ConfigParser()
config.read("config.ini")


class TestDataMaker(unittest.TestCase):

    def setUp(self) -> None:
        self.data_maker = DataMaker()
        
        

    def test_get_data(self):
        self.assertEqual(self.data_maker.get_data(), True)

    def test_split_data(self):
        self.assertEqual(self.data_maker.split_data(), True)

    def test_save_splitted_data(self):
        self.assertEqual(self.data_maker.save_splitted_data(pd.read_csv(
            config["DATA"]["x_data"], index_col=0), config["DATA"]["x_data"]), True)
            
    # def test_zip_extraction_error(self):
    #     """Проверяет, что при ошибке ZipFile вызывается log.error"""
    #     with open(self.data_maker.zip_path, 'w') as f:
    #         f.write("fake zip content")
    #     with patch('zipfile.ZipFile', side_effect=Exception("Broken zip")):
    #         self.data_maker._check_and_extract_zip()
    #         self.assertTrue(True)
            
    # def test_no_zip_file_warning(self):
    #     """Проверяет, что при отсутствии zip-файла выводится предупреждение"""
    #     if os.path.exists(self.data_maker.zip_path):
    #         os.remove(self.data_maker.zip_path)
    #     if os.path.exists(self.data_maker.data_path):
    #         os.remove(self.data_maker.data_path)
    #     self.data_maker._check_and_extract_zip()
    #     self.assertFalse(os.path.exists(self.data_maker.data_path))
    
    # def test_get_data_failure(self):
    #     """Проверяет, что при ошибке сохранения возвращается False"""

    #     with patch('pandas.DataFrame.to_csv', side_effect=OSError("Permission denied")):
    #         result = self.data_maker.get_data()
    #         self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
