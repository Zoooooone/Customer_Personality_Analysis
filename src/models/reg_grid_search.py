import argparse
from Regression import RegressionModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_grid_search_res(args):
    str_to_model_name = {
        "random forest": RandomForestRegressor,
        "lgbm": LGBMRegressor,
        "xgboost": XGBRegressor,
        "knn": KNeighborsRegressor,
        "svm": SVR,
        "linear": LinearRegression
    }
    model_name = str_to_model_name[args.model]
    model = RegressionModel(model_name)

    if model.model_name == RandomForestRegressor:
        params = {
            "n_estimators": [200, 500, 1000],
            "min_samples_split": [2, 3, 4],
        }
        model.grid_search(params)

    if model_name == LGBMRegressor:
        params = {
            "n_estimators": [200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1]
        }
        model.grid_search(params)

    if model_name == XGBRegressor:
        params = {
            "n_estimators": [200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1]
        }
        model.grid_search(params)

    if model_name == KNeighborsRegressor:
        params = {
            "n_neighbors": [3, 4, 5, 6],
            "weights": ["uniform", "distance"],
            "leaf_size": [20, 30, 40]
        }
        model.grid_search(params)

    if model_name == SVR:
        params = {
            "C": [1, 10, 100],
            "kernel": ["linear", "rbf"],
        }
        model.grid_search(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression.py")
    parser.add_argument(
        "--model", type=str, choices=["random forest", "lgbm", "xgboost", "knn", "svm", "linear"]
    )
    args = parser.parse_args()
    
    get_grid_search_res(args)
