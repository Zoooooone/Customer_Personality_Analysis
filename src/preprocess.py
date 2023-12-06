import numpy as np
import pandas as pd


df = pd.read_csv('data/marketing_campaign.csv', sep="\t")  # 'ID'列を削除

df = df.drop('ID', axis=1)
df = df.dropna()
df['Total_Spent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

sign_up_date = df["Dt_Customer"]
sign_up_date = list(map(lambda x: x.split("-")[1:], sign_up_date))
sign_up_date = list(map(lambda x: [int(x[1]), int(x[0])], sign_up_date)) 
months = [(2016 - date[0]) * 12 + (12 - date[1]) for date in sign_up_date]
df.insert(9, "months", months)

df['Total_Spent_Per_Month'] = (df['Total_Spent'] * df['months']) / 24

df['TotalChildren'] = df['Kidhome'] + df['Teenhome']
df['HasChildren'] = df['TotalChildren'] > 0

df = df[df["Year_Birth"] > 1900]
df = df[df["Income"] < 500000]

birth_years = df["Year_Birth"]
birth_years = pd.array(birth_years)
ages = 2016 - birth_years
df.insert(2, "ages", ages)

set(df["Education"])  # {'2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'}
education = df["Education"]
dummy = dict()
for status in set(df["Education"]):
    dummy[status] = [1 if x == status else 0 for x in education]
# visualization
for status, result in dummy.items():
    df.insert(4, status, result)

set(df["Marital_Status"])  # {'Absurd', 'Alone', 'Divorced', 'Married','Single', 'Together', 'Widow', 'YOLO'}
marital_status = df["Marital_Status"]
single = [1 if status in {"Alone", "Divorced", "Single", "Widow"} else 0 for status in marital_status]
married = [1 if status in {"Married", "Together"} else 0 for status in marital_status]
df.insert(10, "single", single)
df.insert(11, "married", married)

df.to_csv("data/marketing_data_preprocess.csv")
