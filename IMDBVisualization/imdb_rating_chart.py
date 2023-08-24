import pandas as pd
from matplotlib import pyplot as plt

# making data frame from csv file
df_imdb = pd.read_csv('imdb_1000.csv')
print("Correlation:", df_imdb['star_rating'].corr(df_imdb['duration']))

df_rating_list = pd.DataFrame({
    'Rating Range': ['0-5', '5-7', '7-8', '8-9', '9-10'],
    'Number of Movies': [0,0,0,0,0]
})
for idx,rate in enumerate(df_imdb['star_rating']):
    if(rate < 5):
        df_rating_list.at[0,'Number of Movies'] += 1
    if (rate >5 and rate<=7):
        df_rating_list.at[1,'Number of Movies'] += 1
    if (rate> 7) and (rate <= 8):
        df_rating_list.at[2,'Number of Movies'] += 1
    if (rate > 8) and (rate <= 9):
        df_rating_list.at[3,'Number of Movies'] += 1
    if (rate> 9):
        df_rating_list.at[4,'Number of Movies'] += 1

print(df_rating_list)

df_rating_list.plot(x="Rating Range", y='Number of Movies', kind='bar')
plt.xlabel("Rating Range")
plt.ylabel("Number of Movies")
plt.title("Rating & Duration Chart")
plt.show()