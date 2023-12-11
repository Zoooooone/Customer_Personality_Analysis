import sys
import pandas as pd
from xgboost import XGBRegressor
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

xgb = XGBRegressor(random_state=42)
xgb_param_grid = {
    'n_estimators': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search(xgb, xgb_param_grid, X, y)

'''
{'learning_rate': 0.01, 'n_estimators': 500}
0.009128340788864474          0.8377257085742162  
'''
