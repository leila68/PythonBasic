from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import re

# Setting up session
s = requests.session()

# List containing all the films for which data has to be scraped from IMDB
films = []

# Lists containing web scraped data
names = []
years = []
ratings = []
genres = []

# Define path where your films are present
# For eg: "/Users/utkarsh/Desktop/films"
# path = input("Enter the path where your films are: ")
path = "D:\github\PythonBasic\IMDBVisualization"

# Films with extensions
filmswe = os.listdir(path)

for film in filmswe:
    # Append into my films list (without extensions)
    films.append(os.path.splitext(film)[0])

df_imdb = pd.read_csv('imdb_1000Copy.csv')
for idx, title in enumerate(df_imdb['title']):
    title = title.lower()

    query = "+".join(title.split())
    URL = "https://www.imdb.com/search/title/?title=" + query
    print(URL)
    try:
        response = s.get(URL)

        # getting content from IMDB Website
        content = response.content
        # print(response.status_code)

        soup = BeautifulSoup(response.content, features="html.parser")

        # searching all films containers found
        containers = soup.find_all("div", class_="lister-item-content")
        for result in containers:
            name1 = result.h3.a.text
            name = result.h3.a.text.lower()
            # Uncomment below lines if you want year specific as well, define year variable before this
            year = result.h3.find("span", class_="lister-item-year text-muted unbold").text

            # if film found (searching using name)
            if title == name:
                # scraping rating
                rating = result.find("div", class_="inline-block ratings-imdb-rating")["data-value"]
                # scraping genre
                genre = result.p.find("span", class_="genre")
                genre = genre.contents[0]

                # appending name, rating and genre to individual lists
                names.append(name1)
                genres.append(genre.strip())
                ratings.append(rating)
                years.append(re.sub("\D", "", year))

    except Exception:
        print("Try again with valid combination of title and release year")

# storing in pandas dataframe
df = pd.DataFrame({'Film Name': names, 'Rating': ratings, 'Genre': genres, 'Year': years})

# making csv using pandas
df.to_csv('film_ratings.csv', index=False, encoding='utf-8')