import configparser
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import shutil
import time
import traceback
import yaml


from src.logger import Logger

SHOW_LOG = True


class Predictor():

    def __init__(self, model: str = "LOG_REG", test_type: str = "func") -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.model_name = model
        try:
            self.model = pickle.load(open(self.config[model]["path"], "rb"))
        except FileNotFoundError as e:
            self.log.error(traceback.format_exc())
            raise RuntimeError(f"Model file not found: {self.config[model]['path']}") from e
        self.test_type = test_type

        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
        self.log.info("Predictor is ready")

    def predict(self) -> tuple[str, float]:
        classifier = self.model
        self.log.info(f"Загружена модель {self.model_name} из {self.config[self.model_name]['path']}")
        scores = {}
        if self.test_type == "smoke":
            try:
                y_pred = classifier.predict(self.X_test)
                y_true = self.y_test.values.ravel()
                
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                scores = {
                    "accuracy": acc,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall
                }
                
                self.log.info(f'{self.config[self.model_name]["path"]} прошел smoke тест')
                return self.model_name, acc
            except Exception as e:
                self.log.error(traceback.format_exc())
                raise RuntimeError(f"Smoke test failed: {e}") from e

        elif self.test_type == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            if not os.path.exists(tests_path):
                raise RuntimeError(f"Tests directory not found: {tests_path}")
            test_files = os.listdir(tests_path)
            if not test_files:
                raise RuntimeError("No test files found in tests directory")
            y_true_array = []
            y_pred_array = []
            acc_array = []
            f1_array = []
            for test in test_files:
                test_file_path = os.path.join(tests_path, test)
                with open(test_file_path) as f:
                    try:
                        data = json.load(f)
                        X = self.sc.transform(pd.json_normalize(data, record_path=['X']).values)
                        y_true = pd.json_normalize(data, record_path=['y']).iloc[:, 0].astype(int).values
                        y_pred = classifier.predict(X)
                        y_true_array.append(y_true)
                        y_pred_array.append(y_pred)
                        
                    except Exception as e:
                        self.log.error(traceback.format_exc())
                        raise RuntimeError(f"Functional test failed for {test}: {e}") from e

                    exp_data = {
                        "model": self.model_name,
                        "model params": dict(self.config.items(self.model_name)),
                        "tests": self.test_type,
                        "X_test path": self.config["SPLIT_DATA"]["X_test"],
                        "y_test path": self.config["SPLIT_DATA"]["y_test"],
                    }
                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir, "exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir, "exp_logfile.log"))
                    shutil.copy(self.config[self.model_name]["path"], os.path.join(exp_dir, f'exp_{self.model_name}.sav'))
            acc = accuracy_score(y_true_array, y_pred_array)
            return self.model_name, float(acc)

        else:
            self.log.error(f'Unknown test type: {self.test_type}')
            raise ValueError(f"Unknown test type: {self.test_type}")

if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()