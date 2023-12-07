import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def plot_feature_importance(y, threshold):
    fig = plt.figure(figsize=(16, 8))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    plt.barh(X_train.columns, feature_importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Barh Chart')
    plt.savefig(f"results/feature_importance_{threshold}.png")


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

for i, y_test in enumerate(y):
    plot_feature_importance(y_test, thresholds[i])
