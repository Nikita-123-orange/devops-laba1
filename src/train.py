import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import traceback
from src.logger import Logger

SHOW_LOG = True


class Model():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        # корень проекта – место, где лежит config.ini
        self.root_dir = os.getcwd()

        # преобразуем относительные пути в абсолютные
        x_train_path = os.path.join(self.root_dir, self.config["SPLIT_DATA"]["x_train"])
        y_train_path = os.path.join(self.root_dir, self.config["SPLIT_DATA"]["y_train"])
        x_test_path = os.path.join(self.root_dir, self.config["SPLIT_DATA"]["x_test"])
        y_test_path = os.path.join(self.root_dir, self.config["SPLIT_DATA"]["y_test"])

        self.X_train = pd.read_csv(x_train_path, index_col=0)
        self.y_train = pd.read_csv(y_train_path, index_col=0)
        self.X_test = pd.read_csv(x_test_path, index_col=0)
        self.y_test = pd.read_csv(y_test_path, index_col=0)

        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

        self.project_path = os.path.join(self.root_dir, "experiments")
        os.makedirs(self.project_path, exist_ok=True)
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.log.info(f"{self.__class__.__name__} is ready")

    def log_reg(self, predict=False) -> bool:
        classifier = LogisticRegression(max_iter=100, solver='saga')
        try:
            self.log.info("Training Logistic Regression model...")
            classifier.fit(self.X_train, self.y_train.values.ravel())
            self.log.info("Logistic Regression model trained successfully")
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if predict:
            y_pred = classifier.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            self.log.info(f"Logistic Regression accuracy: {accuracy}")
            self.log.info(f"Logistic Regression F1 score: {f1}")
            self.log.info(f"Logistic Regression precision: {precision}")
            self.log.info(f"Logistic Regression recall: {recall}")

        # сохраняем относительный путь в конфиг (относительно root_dir)
        rel_model_path = os.path.relpath(self.log_reg_path, start=self.root_dir)
        params = {'path': rel_model_path}
        return self.save_model(classifier=classifier, path=self.log_reg_path, name="LOG_REG", params=params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        # не удаляем config.ini, а перезаписываем
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        self.log.info(f"{name} model saved to {path}")
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = Model()
    multi_model.log_reg(predict=True)