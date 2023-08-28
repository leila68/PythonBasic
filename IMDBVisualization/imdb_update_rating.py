import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# making data frame from csv file
df_imdb = pd.read_csv('imdb_1000.csv')
print("Correlation:", df_imdb['star_rating'].corr(df_imdb['duration']))

sns.lmplot(x='star_rating', y='duration', data=df_imdb)
plt.show()
# <seaborn.axisgrid.FacetGrid at 0xbb8b470>


df_imdb = pd.read_csv('imdb_1000.csv')
for idx, rate in enumerate(df_imdb['title']):
    title = rate.lower()
    query = "+".join(title.split())
    URL = "https://www.imdb.com/search/title/?title=" + query
    print(URL)