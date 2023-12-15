import argparse
from Classification import ClassificationModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def get_grid_search_res(args):
    str_to_model_name = {
        "random forest": RandomForestClassifier,
        "lgbm": LGBMClassifier,
        "xgboost": XGBClassifier,
    }
    model_name = str_to_model_name[args.model]
    model = ClassificationModel(model_name)

    if model.model_name == RandomForestClassifier:
        params = {
            "n_estimators": [200, 500, 1000],
            "min_samples_split": [2, 3, 4],
        }
        model.grid_search(params)

    if model_name == LGBMClassifier:
        params = {
            "n_estimators": [200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1]
        }
        model.grid_search(params)

    if model_name == XGBClassifier:
        params = {
            "n_estimators": [200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1]
        }
        model.grid_search(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression.py")
    parser.add_argument(
        "--model", type=str, choices=["random forest", "lgbm", "xgboost"]
    )
    args = parser.parse_args()

    get_grid_search_res(args)
