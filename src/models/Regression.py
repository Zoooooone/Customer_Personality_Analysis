import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


class RegressionModel:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        if self.model_name == RandomForestRegressor or self.model_name == XGBRegressor:
            self.model = self.model_name(random_state=42, **kwargs)
        elif self.model_name == LGBMRegressor:
            self.model = self.model_name(random_state=42, verbose=-1, **kwargs)
        else:
            self.model = self.model_name(**kwargs)
        self.model_names = {
            RandomForestRegressor: "random forest",
            LGBMRegressor: "lgbm",
            XGBRegressor: "xgboost",
            SVR: "svm",
            LinearRegression: "linear",
            KNeighborsRegressor: "knn"
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
            # "NumDealsPurchases",
            "NumWebVisitsMonth", 
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
            "NwpPerVM", 
            "NwpPerV"
        ]
        self.X = self.df[self.columns]
        self.y = self.df["WebRatio"]

    def cross_validation(self):
        def root_mean_square_error(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))
        
        metrics = [r2_score, mean_squared_error, root_mean_square_error]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        res = []
        for metric in metrics:
            res.append(cross_val_score(self.model, self.X, self.y, scoring=make_scorer(metric), cv=kf))
        res = list(map(np.mean, res))
        return res
    
    def get_res(self):
        res = defaultdict(list)

        r2, mse, rmse = self.cross_validation()
        res["model"].append(self.model_names[self.model_name])
        res["r2"].append(r2)
        res["mse"].append(mse)
        res["rmse"].append(rmse)
        
        res = pd.DataFrame(res)
        output_path = "results/regression_res.csv"
        if os.path.exists(output_path):
            res.to_csv(output_path, header=False, mode="a")
        else:
            res.to_csv(output_path, header=True, mode="w")

    def grid_search(self, param_grid):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        model_cv = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, verbose=1)
        model_cv.fit(X_train, y_train)

        best_params = model_cv.best_params_
        best_model = model_cv.best_estimator_
        y_pred = best_model.predict(X_test)

        res = defaultdict(list)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n" + self.model_names[self.model_name])
        print(best_params)
        print(f"{mse:<30}{r2:<30}\n")
        
        res["model"].append(self.model_names[self.model_name])
        res["best_params"].append(best_params)
        res["r2"].append(r2)
        res["mse"].append(mse)

        res = pd.DataFrame(res)
        output_path = "results/grid_search_reg.csv"
        if os.path.exists(output_path):
            res.to_csv(output_path, header=False, mode="a")
        else:
            res.to_csv(output_path, header=True, mode="w")


if __name__ == "__main__":
    random_forest = RegressionModel(
        RandomForestRegressor,
        n_estimators=1000,
        min_samples_split=3
    )

    lgbm = RegressionModel(
        LGBMRegressor,
        n_estimators=500,
        learning_rate=0.05
    )

    xgboost = RegressionModel(
        XGBRegressor,
        n_estimators=500,
        learning_rate=0.1
    )

    random_forest.get_res()
    lgbm.get_res()
    xgboost.get_res()
