# src/models/randomForest.py
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
sys.path.append("src/models/")
from reg_model_test import grid_search

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
X, y = df[columns], df["WebRatio"]

rf = RandomForestRegressor(random_state=42)
rf_param_grid = {
    "n_estimators": [200, 500, 1000],
    "min_samples_split": [2, 3, 4],
}

grid_search(rf, rf_param_grid, X, y)

'''
{'min_samples_split': 3, 'n_estimators': 1000}
0.007703202685398734          0.8523867955796841
''' 
