import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier


def cross_validation(model, X, y):
    metrics = [precision_score, recall_score, accuracy_score, f1_score]
    res = []
    for metric in metrics:
        res.append(cross_val_score(model, X, y, scoring=make_scorer(metric), cv=5))
    res = list(map(np.mean, res))
    return res


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

    print(cross_validation(RandomForestClassifier(n_estimators=500, random_state=42), X, y_75))
    print(cross_validation(RandomForestClassifier(n_estimators=500, random_state=42), X, y_50))
    print(cross_validation(RandomForestClassifier(n_estimators=500, random_state=42), X, y_25))
