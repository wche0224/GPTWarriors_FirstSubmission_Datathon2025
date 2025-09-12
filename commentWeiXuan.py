import os
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
#from zipFile import zipFile

#run the code below if its your first time using nltk library
#nltk.download('stopwords')

dataFrame1 = pd.read_csv("dataset/comments1.csv")
#print(dataFrame1.head())

valueTypes = dataFrame1["kind"].unique()
#print(valueTypes)
#print(dataFrame1.columns)

#compare textOriginal, likeCount, publishedAt
newDataFrame1 = dataFrame1[["textOriginal", "likeCount", "publishedAt"]]
filteredLikesDF = newDataFrame1[newDataFrame1["likeCount"] > 0]
#print(newDataFrame1.head())

averageLikeCount1 = filteredLikesDF["likeCount"].mean()
print(averageLikeCount1)    
successfulComments1 = newDataFrame1[newDataFrame1["likeCount"] >= averageLikeCount1]
print(type(successfulComments1["textOriginal"]))

def textOriginalAnalysis(df) -> dict:
    dfT = df["textOriginal"] #dfT short for dfText
    textArray = []
    stop_words = set(stopwords.words('english'))
    for i in dfT:  
        # ?:^ checks for if it is the start, can accept no whitespace if at the start
        words = re.findall(r'(?:^|\s)([1-9][0-9]*|[a-zA-Z]+)(?=\s|$)', i.lower(), flags=re.IGNORECASE)

        filtered_words = [word for word in words if word not in stop_words]

        textArray.extend(filtered_words)

    frequency_dict = Counter(textArray)
    topTenPercent = int(len(frequency_dict) / 10)

    return frequency_dict.most_common(topTenPercent)


analyzed = textOriginalAnalysis(successfulComments1)
print(analyzed)

print(filteredLikesDF["publishedAt"].tail())
print(filteredLikesDF["publishedAt"].dtype)


