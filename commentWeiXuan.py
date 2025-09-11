import os
import pandas as pd
#from zipFile import zipFile

dataFrame1 = pd.read_csv("dataset/comments1.csv")
#print(dataFrame1.head())

valueTypes = dataFrame1["kind"].unique()
#print(valueTypes)
#print(dataFrame1.columns)

#compare textOriginal, likeCount, publishedAt
newDataFrame1 = dataFrame1[["textOriginal", "likeCount", "publishedAt"]]
print(newDataFrame1.head())
averageLikeCount1 = newDataFrame1["likeCount"].mean()
print(averageLikeCount1)    
successfulComments1 = newDataFrame1[newDataFrame1["likeCount"] >= averageLikeCount1]

