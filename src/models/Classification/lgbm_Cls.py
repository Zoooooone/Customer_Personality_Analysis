import sys
import pandas as pd
from lightgbm import LGBMClassifier
sys.path.append("src/")
from cls_model_test import cross_validation
from collections import defaultdict

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
X, y_75, y_50, y_25 = df[columns], df["threshold_75"], df["threshold_50"], df["threshold_25"]
y = [y_25, y_50, y_75]
thresholds = [25, 50, 75]
res = defaultdict(list)

for i in range(len(y)):
    pre, rec, acc, f1 = cross_validation(LGBMClassifier(n_estimators=1000, random_state=42, verbose=-1), X, y[i])
    res["model"].append("lgbm")
    res["threshold"].append(thresholds[i])
    res["precision"].append(pre)
    res["recall"].append(rec)
    res["accuracy"].append(acc)
    res["f1"].append(f1)

res = pd.DataFrame(res)
res.to_csv("results/classification_res.csv", header=False, mode="a")
