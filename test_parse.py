import sys
from bs4 import BeautifulSoup

with open("xmls/29980676.xml", "r") as f:
    soup = BeautifulSoup(f, "lxml-xml")

tables = soup.find_all("table-wrap")
for i, table in enumerate(tables):
    print(f"Table {i+1}:")
    caption = table.find("caption")
    if caption: print("Caption:", caption.text.strip())
    
    headers = []
    for th in table.find_all("th"):
        headers.append(th.text.strip())
    print("Headers:", headers)
