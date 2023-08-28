# beauty_soup.py

from urllib.request import urlopen
from bs4 import BeautifulSoup

# Using Beautiful Soup, print out a list of all the links on the page by looking for
# HTML tags with the name a and retrieving the value taken on by the href attribute of each tag.

base_url = "https://www.imdb.com/"
html_page = urlopen(base_url + "/search/title/?count=100&groups=top_1000&sort=user_rating")
html_text = html_page.read().decode("utf-8")
soup = BeautifulSoup(html_text, "html.parser")
for link in soup.find_all("a"):
    link_url = base_url + link["href"]
    print(link_url)