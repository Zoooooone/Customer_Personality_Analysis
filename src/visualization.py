import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import Counter

column_name_to_category = {
    "MntWines": "wine", 
    "MntFruits": "fruits", 
    "MntMeatProducts": "meat",
    "MntFishProducts": "fish",
    "MntSweetProducts": "sweet",
    "MntGoldProds": "gold"
}


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
    fig = plt.figure(figsize=(14, 14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig("img/heatmap.png")


def plot_cls_res(metric):
    data = pd.read_csv("results/classification_res.csv")
    x = np.arange(3)
    rf = data[data["model"] == "random forest"][metric]
    lgbm = data[data["model"] == "lgbm"][metric]
    xgboost = data[data["model"] == "xgboost"][metric]

    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    categories = ["25%", "50%", "75%"]

    fig = plt.figure(figsize=(8, 6))
    plt.bar(x, rf, width=width, label='random forest')
    plt.bar(x + width, lgbm, width=width, label='lgbm')
    plt.bar(x + 2 * width, xgboost, width=width, label='xgboost')
    plt.xticks(x + width, categories)
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("thresholds")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"results/{metric}.png")


def plot_web_purchase_over_median_pie():
    categories = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    purchase = df[df["WebRatio"] >= df["WebRatio"].describe().at["50%"]][categories]
    purchase_sum = dict()
    for column in purchase.columns:
        purchase_sum[column_name_to_category[column]] = purchase[column].sum()
    labels, counts = zip(*purchase_sum.items())

    fig1 = plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=None, autopct='%d%%', startangle=90, pctdistance=0.75, textprops={'fontsize': 12})
    plt.legend(labels, loc="best")
    plt.title("purchase (web ratio >= 50%)")
    plt.savefig("results/web_purchase_over_median_pie.png")

    categories_over_median = df[df["WebRatio"] >= df["WebRatio"].describe().at["50%"]]["most_purchased"]
    counter = Counter(categories_over_median)
    labels, counts = zip(*counter.items())

    fig2 = plt.figure(figsize=(8, 6), dpi=400)
    plt.pie(counts, labels=None, autopct='%d%%', startangle=90, pctdistance=0.75, textprops={'fontsize': 12})
    plt.legend(labels, loc="best")
    plt.title("most purchased categories (web ratio >= 50%)")
    plt.savefig("results/most_purchased_categories_over_median_pie.png")


def plot_web_purchase_under_median_pie():
    categories = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    purchase = df[df["WebRatio"] < df["WebRatio"].describe().at["50%"]][categories]
    purchase_sum = dict()
    for column in purchase.columns:
        purchase_sum[column_name_to_category[column]] = purchase[column].sum()
    labels, counts = zip(*purchase_sum.items())

    fig1 = plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=None, autopct='%d%%', startangle=90, pctdistance=0.75, textprops={'fontsize': 12})
    plt.legend(labels, loc="best")
    plt.title("purchase (web ratio < 50%)")
    plt.savefig("results/web_purchase_under_median_pie.png")

    categories_over_median = df[df["WebRatio"] < df["WebRatio"].describe().at["50%"]]["most_purchased"]
    counter = Counter(categories_over_median)
    labels, counts = zip(*counter.items())

    fig = plt.figure(figsize=(8, 6), dpi=400)
    plt.pie(counts, labels=None, autopct='%d%%', startangle=90, pctdistance=0.85, textprops={'fontsize': 8})
    plt.legend(labels, loc="best")
    plt.title("most purchased categories (web ratio < 50%)")
    plt.savefig("results/most_purchased_categories_under_median_pie.png")


def plot_web_purchase_pie():
    categories = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    purchase = df[categories]
    purchase_sum = dict()
    for column in purchase.columns:
        purchase_sum[column_name_to_category[column]] = purchase[column].sum()
    labels, counts = zip(*purchase_sum.items())

    fig1 = plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=None, autopct='%d%%', startangle=90, pctdistance=0.75, textprops={'fontsize': 12})
    plt.legend(labels, loc="best")
    plt.title("purchase")
    plt.savefig("results/web_purchase_pie.png")

    categories_over_median = df["most_purchased"]
    counter = Counter(categories_over_median)
    labels, counts = zip(*counter.items())

    fig = plt.figure(figsize=(8, 6), dpi=400)
    plt.pie(counts, labels=None, autopct='%d%%', startangle=90, pctdistance=0.85, textprops={'fontsize': 8})
    plt.legend(labels, loc="best")
    plt.title("most purchased categories")
    plt.savefig("results/most_purchased_categories_pie.png")


def plot_web_purchase_over_median_bar():
    categories = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    purchase = df[df["WebRatio"] >= df["WebRatio"].describe().at["50%"]][categories]
    purchase_sum = dict()
    for column in purchase.columns:
        purchase_sum[column_name_to_category[column]] = purchase[column].sum()
    labels, counts = zip(*purchase_sum.items())

    fig1 = plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title("purchase (web ratio >= 50%)")
    plt.xlabel("categories")
    plt.ylabel("the sum of consumption")
    plt.savefig("results/web_purchase_over_median_bar.png")

    categories_over_median = df[df["WebRatio"] >= df["WebRatio"].describe().at["50%"]]["most_purchased"]
    counter = Counter(categories_over_median)
    labels, counts = zip(*counter.items())

    fig2 = plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title("most purchased categories (web ratio >= 50%)")
    plt.xlabel("categories")
    plt.ylabel("number of people")
    plt.savefig("results/most_purchased_categories_over_median_bar.png")


def plot_web_purchase_under_median_bar():
    categories = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    purchase = df[df["WebRatio"] < df["WebRatio"].describe().at["50%"]][categories]
    purchase_sum = dict()
    for column in purchase.columns:
        purchase_sum[column_name_to_category[column]] = purchase[column].sum()
    labels, counts = zip(*purchase_sum.items())

    fig1 = plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title("purchase (web ratio < 50%)")
    plt.xlabel("categories")
    plt.ylabel("the sum of consumption")
    plt.savefig("results/web_purchase_under_median_bar.png")

    categories_over_median = df[df["WebRatio"] < df["WebRatio"].describe().at["50%"]]["most_purchased"]
    counter = Counter(categories_over_median)
    labels, counts = zip(*counter.items())

    fig2 = plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title("most purchased categories (web ratio < 50%)")
    plt.xlabel("categories")
    plt.ylabel("number of people")
    plt.savefig("results/most_purchased_categories_under_median_bar.png")


def plot_web_purchase_bar():
    categories = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    purchase = df[categories]
    purchase_sum = dict()
    for column in purchase.columns:
        purchase_sum[column_name_to_category[column]] = purchase[column].sum()
    labels, counts = zip(*purchase_sum.items())

    fig1 = plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title("purchase")
    plt.xlabel("categories")
    plt.ylabel("the sum of consumption")
    plt.savefig("results/web_purchase__bar.png")

    categories_over_median = df["most_purchased"]
    counter = Counter(categories_over_median)
    labels, counts = zip(*counter.items())

    fig2 = plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title("most purchased categories")
    plt.xlabel("categories")
    plt.ylabel("number of people")
    plt.savefig("results/most_purchased_categories_bar.png")


def plot_web_purchase_all():
    plot_web_purchase_over_median_pie()
    plot_web_purchase_under_median_pie()
    plot_web_purchase_pie()
    plot_web_purchase_over_median_bar()
    plot_web_purchase_under_median_bar()
    plot_web_purchase_bar()


if __name__ == "__main__":
    df = pd.read_csv("data/marketing_data_preprocess.csv")
    # print(df.columns)

    """
    Index(['Unnamed: 0', 'Year_Birth', 'Education', 'ages', 'Marital_Status',
       'Graduation', 'PhD', 'Basic', '2n Cycle', 'Master', 'Income', 'single',
       'married', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines',
       'months', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
       'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact',
       'Z_Revenue', 'Response', 'Total_Spent', 'Total_Spent_Per_Month',
       'Childbin', 'Kidbin', 'Teenbin', 'NwpPerVM', 'NwpPerV', 'NumPurchases',
       'WebRatio'],
      dtype='object')
    """
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
        "NwpPerV",
        "Total_Spent", 
        "Total_Spent_Per_Month", 
        "Childbin", 
        "Kidbin", 
        "Teenbin",
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
        "NwpPerV",
        "Total_Spent", 
        "Total_Spent_Per_Month", 
        "Childbin", 
        "Kidbin", 
        "Teenbin",
        "WebRatio"
    ]
    
    # plot_hist(selected_columns)
    # pair_plot(selected_columns)
    # heatmap(selected_columns_heatmap)

    metrics = ["precision", "recall", "accuracy", "f1"]
    for metric in metrics:
        plot_cls_res(metric)

    plot_web_purchase_all()
