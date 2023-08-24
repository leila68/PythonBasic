import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# making data frame from csv file
df_imdb = pd.read_csv('imdb_1000.csv')

actors_list = []
for i, ac in enumerate(df_imdb['actors_list']):
    temp = df_imdb.at[i,'actors_list'].split(',')
    actors_list.append(temp)

pair_index_list = []
for i1, ac1 in enumerate(actors_list):
    for i2 in ac1:
        for j1, ac2 in enumerate(actors_list[i1+1:], i1+1):
            for j2 in ac2:
                if i2 == j2:
                    pair_index_list.append((i1, j1))
print(pair_index_list)

g = nx.Graph()
g.add_edges_from(pair_index_list)
nx.draw(g, with_labels=True)
plt.savefig("filename.png")