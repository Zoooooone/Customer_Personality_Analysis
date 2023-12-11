import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

column_name_to_category = {
    "MntWines": "wine", 
    "MntFruits": "fruits", 
    "MntMeatProducts": "meat",
    "MntFishProducts": "fish",
    "MntSweetProducts": "sweet",
    "MntGoldProds": "gold"
}


def plot_hist(df):
    def smallest_square_number(n):
        res = 1
        while res ** 2 < n:
            res += 1
        return res

    columns = [
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
    n = len(columns)
    row = smallest_square_number(n)
    fig, axes = plt.subplots(row, row, figsize=(25, 25))
    for i in tqdm(range(n)):
        sns.histplot(df[columns[i]], bins=15, ax=axes[i // row, i % row])
    plt.savefig("img/histogram.png")


def pair_plot(df):
    columns = [
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
    sns.pairplot(df[columns])
    plt.savefig("img/pairplot.png")


def plot_heatmap(df):
    columns = [
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


def plot_web_purchase_over_median_pie(df, args):
    categories = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

    if args.relation_with_median == "over":
        purchase = df[df["WebRatio"] >= df["WebRatio"].describe().at["50%"]][categories]
        categories_over_median = df[df["WebRatio"] >= df["WebRatio"].describe().at["50%"]]["most_purchased"]
    elif args.relation_with_median == "under":
        purchase = df[df["WebRatio"] < df["WebRatio"].describe().at["50%"]][categories]
        categories_over_median = df[df["WebRatio"] < df["WebRatio"].describe().at["50%"]]["most_purchased"]
    else:
        purchase = df[categories]
        categories_over_median = df["most_purchased"]

    purchase_sum = dict()
    for column in categories:
        purchase_sum[column_name_to_category[column]] = sum(purchase[column])

    counter = Counter(categories_over_median)
    purchase_sum = dict(sorted(purchase_sum.items(), key=lambda x: x[0]))
    counter = dict(sorted(counter.items(), key=lambda x: x[0]))

    graph = args.graph

    def set_title(title):
        if title == "amount of purchase":
            if args.relation_with_median == "over":
                plt.title("amount of purchase (web ratio >= 50%)")
            elif args.relation_with_median == "under":
                plt.title("amount of purchase (web ratio < 50%)")
            else:
                plt.title("amount of purchase")
        else:
            if args.relation_with_median == "over":
                plt.title("most purchased category (web ratio >= 50%)")
            elif args.relation_with_median == "under":
                plt.title("most purchased category (web ratio < 50%)")
            else:
                plt.title("most purchased category")

    if graph == "pie":
        labels, counts = zip(*purchase_sum.items())
        logger.info(f"{' amount of purchase':<30}{str(labels):<60}{str(counts):<60}")
        
        fig1 = plt.figure(figsize=(10, 8))
        plt.pie(counts, labels=None, autopct='%d%%', startangle=90, counterclock=False, pctdistance=0.8, textprops={'fontsize': 10})
        plt.legend(labels, loc="best")
        set_title("amount of purchase")
        plt.savefig(f"results/{graph}/web_purchase_{args.relation_with_median}_median_{graph}.png")

        labels, counts = zip(*counter.items())
        logger.info(f"{' most purchased category':<30}{str(labels):<60}{str(counts):<60}")

        fig2 = plt.figure(figsize=(10, 8), dpi=400)
        plt.pie(counts, labels=None, autopct='%d%%', startangle=90, counterclock=False, pctdistance=0.8, textprops={'fontsize': 10})
        plt.legend(labels, loc="best")
        set_title("most purchased category")
        plt.savefig(f"results/{graph}/most_purchased_categories_{args.relation_with_median}_median_{graph}.png")
    
    else:
        labels, counts = zip(*purchase_sum.items())

        fig1 = plt.figure(figsize=(10, 8))
        plt.bar(labels, counts)
        plt.xlabel("categories")
        plt.ylabel("the sum of consumption")
        set_title("amount of purchase")
        plt.savefig(f"results/{graph}/web_purchase_{args.relation_with_median}_median_{graph}.png")

        labels, counts = zip(*counter.items())

        fig2 = plt.figure(figsize=(10, 8))
        plt.bar(labels, counts)
        plt.xlabel("categories")
        plt.ylabel("number of people")
        set_title("most purchased category")
        plt.savefig(f"results/{graph}/most_purchased_categories_{args.relation_with_median}_median_{graph}.png")


def main(args):
    df = pd.read_csv("data/marketing_data_preprocess.csv")
    
    if args.plot == "histogram":
        plot_hist(df)

    if args.plot == "pairplot":
        pair_plot(df)
    
    if args.plot == "heatmap":
        plot_heatmap(df)

    if args.plot == "classification_metrics":
        metrics = ["precision", "recall", "accuracy", "f1"]
        for metric in metrics:
            plot_cls_res(metric)

    if args.plot == "web_purchase":
        plot_web_purchase_over_median_pie(df, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualization.py")
    parser.add_argument(
        "--plot", type=str, choices=["histogram", "pairplot", "heatmap", "classification_metrics", "web_purchase"]
    )
    parser.add_argument(
        "--relation_with_median", type=str, choices=["over", "under", "whole"]
    )
    parser.add_argument(
        "--graph", type=str, choices=["pie", "bar"]
    )

    args = parser.parse_args()
    main(args)
