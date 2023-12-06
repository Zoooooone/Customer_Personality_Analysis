import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

df = pd.read_csv("data/data_final.csv")
# print(df.columns)

"""
Index(['Unnamed: 0', 'Year_Birth', 'Education', 'ages', 'Marital_Status',
       'Graduation', 'Master', '2n Cycle', 'Basic', 'PhD', 'Income', 'single',
       'married', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines',
       'months', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
       'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact',
       'Z_Revenue', 'Response', 'Total_Spent', 'Total_Spent_Per_Month',
       'TotalChildren', 'HasChildren'],
      dtype='object')
"""


def plot_hist(columns):
    def smallest_square_number(n):
        res = 1
        while res ** 2 < n:
            res += 1
        return res

    n = len(columns)
    row = smallest_square_number(n)
    fig, axes = plt.subplots(row, row, figsize=(25, 25))
    for i in tqdm(range(n)):
        sns.histplot(df[columns[i]], bins=15, ax=axes[i // row, i % row])
    plt.savefig("img/histogram.png")


def pair_plot(columns):
    sns.pairplot(df[columns])
    plt.savefig("img/pairplot.png")


def heatmap(columns):
    correlation_matrix = df[columns].corr()
    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig("img/heatmap.png")


if __name__ == "__main__":
    selected_columns = [
        "ages", 
        "Education", 
        "Income", 
        "months", 
        "Recency", 
        "MntWines", 
        "MntFruits", 
        "MntMeatProducts", 
        "MntFishProducts", 
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "WebRatio"
    ]

    selected_columns_heatmap = [
        "ages",  
        "Income", 
        "months", 
        "Recency", 
        "MntWines", 
        "MntFruits", 
        "MntMeatProducts", 
        "MntFishProducts", 
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "WebRatio"
    ]
    
    plot_hist(selected_columns)
    pair_plot(selected_columns)
    heatmap(selected_columns_heatmap)
