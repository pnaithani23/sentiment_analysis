#Reading the csv
import pandas as pd
fox_df = pd.read_csv("Fox.csv")
otctz_df = pd.read_csv("otctz.csv")
star_df = pd.read_csv("Thestar.csv")
nyt_df = pd.read_csv("nyt.csv")

#combining all the dataframe into one
description = pd.concat([fox_df, otctz_df, star_df, nyt_df], ignore_index=True)
description.head()

#analyzing sentiments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


emptyline = []
for row in description['summary']:
    ps = analyzer.polarity_scores(row)
    emptyline.append(ps)


#converting list into dataframe
df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()

#Joining description dataframe and sentiments dataframe 
df_join = pd.concat([description.reset_index(drop=True), df_sentiments], axis=1)
df_join.head()


#Creating a new column
df_join['Sentiment'] = df_join['compound'].apply(lambda score: 'positive' if score>=0.01 else 'negative' if score<=-0.01 else 'neutral')
df_join.head()

#exporting dataframe to csv
df_join.to_csv('test.csv', index=False)
