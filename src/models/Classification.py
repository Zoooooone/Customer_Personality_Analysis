import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class ClassificationModel:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        if self.model_name == LGBMClassifier:
            self.model = self.model_name(random_state=42, verbose=-1, **kwargs)
        else:
            self.model = self.model_name(random_state=42, **kwargs)
        self.model_names = {
            RandomForestClassifier: "random forest", 
            LGBMClassifier: "lgbm", 
            XGBClassifier: "xgboost"
        }
        self.df = pd.read_csv("data/marketing_data_preprocess.csv")
        self.columns = [
            "ages", 
            "2n Cycle", 
            "Basic", 
            "Graduation", 
            "Master", 
            "PhD", 
            "single", 
            "married", 
            "Income", 
            "Kidhome", 
            "Teenhome", 
            "months", 
            # "Recency", 
            "MntWines", 
            "MntFruits", 
            "MntMeatProducts", 
            "MntFishProducts", 
            "MntSweetProducts", 
            "MntGoldProds",
            "NumDealsPurchases", 
            # "NumWebVisitsMonth", 
            # "AcceptedCmp3", 
            # "AcceptedCmp4", 
            # "AcceptedCmp5", 
            # "AcceptedCmp1", 
            # "AcceptedCmp2", 
            # "Complain", 
            # "Z_CostContact", 
            # "Z_Revenue", 
            # "Response", 
            "Total_Spent", 
            "Total_Spent_Per_Month", 
            "Childbin", 
            "Kidbin", 
            "Teenbin", 
            # "NwpPerVM", 
            # "NwpPerV"
        ]
        self.X = self.df[self.columns]
        self.y_75, self.y_50, self.y_25 = self.df["threshold_75"], self.df["threshold_50"], self.df["threshold_25"]

    def cross_validation(self, y):
        metrics = [precision_score, recall_score, accuracy_score, f1_score]
        res = []
        for metric in metrics:
            res.append(cross_val_score(self.model, self.X, y, scoring=make_scorer(metric), cv=5))
        res = list(map(np.mean, res))
        return res
    
    def get_res(self, threshold):
        y = {25: self.y_25, 50: self.y_50, 75: self.y_75}
        res = defaultdict(list)

        pre, rec, acc, f1 = self.cross_validation(y[threshold])
        res["model"].append(self.model_names[self.model_name])
        res["threshold"].append(threshold)
        res["precision"].append(pre)
        res["recall"].append(rec)
        res["accuracy"].append(acc)
        res["f1"].append(f1)

        res = pd.DataFrame(res)
        output_path = "results/classification_res.csv"
        if os.path.exists(output_path):
            res.to_csv(output_path, header=False, mode="a")
        else:
            res.to_csv(output_path, header=True, mode="w")

    def grid_search(self, param_grid):
        y = [self.y_25, self.y_50, self.y_75]
        thresholds = [25, 50, 75]
        res = defaultdict(list)

        for i in range(len(y)):
            X_train, X_test, y_train, y_test = train_test_split(self.X, y[i], test_size=0.2)
            model_cv = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, verbose=1)
            model_cv.fit(X_train, y_train)

            best_params = model_cv.best_params_
            best_model = model_cv.best_estimator_
            y_pred = best_model.predict(X_test)

            pre = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print("\n" + self.model_names[self.model_name])
            print(best_params)
            print(f"pre: {pre:<15.4f} rec: {rec:<15.4f} acc: {acc:<15.4f} f1: {f1:<15.4f}\n")

            res["model"].append(self.model_names[self.model_name])
            res["threshold"].append(thresholds[i])
            res["best_params"].append(best_params)
            res["precision"].append(pre)
            res["recall"].append(rec)
            res["accuracy"].append(acc)
            res["f1"].append(f1)

        res = pd.DataFrame(res)
        output_path = "results/grid_search_cls.csv"
        if os.path.exists(output_path):
            res.to_csv(output_path, header=False, mode="a")
        else:
            res.to_csv(output_path, header=True, mode="w")


if __name__ == "__main__":
    random_forest_25 = ClassificationModel(
        RandomForestClassifier, 
        n_estimators=500,
        min_samples_split=2
    )

    random_forest_50 = ClassificationModel(
        RandomForestClassifier, 
        n_estimators=1000,
        min_samples_split=2
    )

    random_forest_75 = ClassificationModel(
        RandomForestClassifier, 
        n_estimators=1000,
        min_samples_split=4
    )

    lgbm_25 = ClassificationModel(
        LGBMClassifier,
        n_estimators=500,
        learning_rate=0.01
    )

    lgbm_50 = ClassificationModel(
        LGBMClassifier,
        n_estimators=500,
        learning_rate=0.01
    )

    lgbm_75 = ClassificationModel(
        LGBMClassifier,
        n_estimators=200,
        learning_rate=0.01
    )

    xgboost_25 = ClassificationModel(
        XGBClassifier,
        n_estimators=200,
        learning_rate=0.1
    )

    xgboost_50 = ClassificationModel(
        XGBClassifier,
        n_estimators=1000,
        learning_rate=0.01
    )

    xgboost_75 = ClassificationModel(
        XGBClassifier,
        n_estimators=200,
        learning_rate=0.1
    )

    random_forest_25.get_res(25)
    random_forest_50.get_res(50)
    random_forest_75.get_res(75)
    lgbm_25.get_res(25)
    lgbm_50.get_res(50)
    lgbm_75.get_res(75)
    xgboost_25.get_res(25)
    xgboost_50.get_res(50)
    xgboost_75.get_res(75)
