import pandas as pd

df = pd.read_csv("data/marketing_data_preprocess.csv")
df = df[["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumPurchases", "WebRatio"]]
df.to_csv("data/web_relation.csv")

df2 = pd.read_csv("data/filtered_data7.csv")
df2 = df2[["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumPurchases", "WebRatio"]]
df2.to_csv("data/web_relation_2.csv")
