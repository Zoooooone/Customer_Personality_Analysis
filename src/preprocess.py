import numpy as np
import pandas as pd


df = pd.read_csv('data/marketing_campaign.csv', sep="\t")  # 'ID'列を削除

# drop 
df = df.drop('ID', axis=1)
df = df.dropna()

# registered months
sign_up_date = df["Dt_Customer"]
sign_up_date = list(map(lambda x: x.split("-")[1:], sign_up_date))
sign_up_date = list(map(lambda x: [int(x[1]), int(x[0])], sign_up_date)) 
months = [(2016 - date[0]) * 12 + (12 - date[1]) for date in sign_up_date]
df.insert(9, "months", months)

# total_spent
df['Total_Spent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['Total_Spent_Per_Month'] = (df['Total_Spent'] * df['months']) / 24

# children
df["Childbin"] = ((df["Kidhome"] > 0) & (df["Teenhome"] > 0)).astype(int)
df["Kidbin"] = (df["Kidhome"] > 0).astype(int)
df["Teenbin"] = (df["Teenhome"] > 0).astype(int)

# age
birth_years = df["Year_Birth"]
birth_years = pd.array(birth_years)
ages = 2016 - birth_years
df.insert(2, "ages", ages)

# education status
set(df["Education"])  # {'2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'}
education = df["Education"]
dummy = dict()
for status in set(df["Education"]):
    dummy[status] = [1 if x == status else 0 for x in education]
for status, result in dummy.items():
    df.insert(4, status, result)

# marital status
set(df["Marital_Status"])  # {'Absurd', 'Alone', 'Divorced', 'Married','Single', 'Together', 'Widow', 'YOLO'}
marital_status = df["Marital_Status"]
single = [1 if status in {"Alone", "Divorced", "Single", "Widow"} else 0 for status in marital_status]
married = [1 if status in {"Married", "Together"} else 0 for status in marital_status]
df.insert(10, "single", single)
df.insert(11, "married", married)

# website purchase number (and its value for per month)
df["NwpPerVM"] = df["NumWebPurchases"] / df["NumWebVisitsMonth"] / df["months"]
df["NwpPerV"] = df["NumWebPurchases"] / df["NumWebVisitsMonth"]
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# purchase number
df["NumPurchases"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]

# website purchase ratio
df["WebRatio"] = df["NumWebPurchases"] / df["NumPurchases"]

# website purchase ratio classified by thresholds
# print(df["WebRatio"].describe())
'''
count    2202.000000
mean        0.330305
std         0.119683
min         0.000000
25%         0.250000
50%         0.333333
75%         0.400000
max         1.000000
'''
threshold_75 = [1 if x >= 0.4 else 0 for x in df["WebRatio"]]
threshold_50 = [1 if x >= 0.333333 else 0 for x in df["WebRatio"]]
threshold_25 = [1 if x >= 0.25 else 0 for x in df["WebRatio"]]
df["threshold_75"] = threshold_75
df["threshold_50"] = threshold_50
df["threshold_25"] = threshold_25

# outliers
df = df[df["Year_Birth"] > 1900]
df = df[df["Income"] < 500000]

# drop nan
df = df.dropna()

# most purchased category from website
purchases = df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]]
df["most_purchased"] = purchases.idxmax(axis=1)


def rename_category(column_name):
    column_name_to_category = {
        "MntWines": "wine", 
        "MntFruits": "fruits", 
        "MntMeatProducts": "meat",
        "MntFishProducts": "fish",
        "MntSweetProducts": "sweet",
        "MntGoldProds": "gold"
    }
    return column_name_to_category[column_name]


df["most_purchased"] = df["most_purchased"].apply(rename_category)

# save the dataframe
df.to_csv("data/marketing_data_preprocess.csv")
