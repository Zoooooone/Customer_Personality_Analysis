import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def cross_validation(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    average_mse = np.mean(cv_results)
    return -average_mse


def model_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Support Vector Machine': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1)
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse_val = mse(y_test, y_pred)
        print(f'{model_name:<30} R2 Score = {r2:.4f}, Mean Squared Error = {mse_val:.4f}')


def grid_search(model, param_grid, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1)
    model_cv.fit(X_train, y_train)

    best_params = model_cv.best_params_
    best_model = model_cv.best_estimator_

    y_pred = best_model.predict(X_test)
    loss = mse(y_test, y_pred)
    score = r2_score(y_test, y_pred)

    print(best_params)
    print(f"{loss:<30}{score:<30}")
    return best_params, best_model, loss, score


if __name__ == "__main__":
    df = pd.read_csv("data/marketing_data_preprocess.csv")
    columns = [
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
    X, y = df[columns], df["WebRatio"]

    model_test(X, y)
    
