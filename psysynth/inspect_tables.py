import os
import re
from bs4 import BeautifulSoup

def clean_text(t):
    return t.lower().replace('-', ' ').replace('/', ' ')

for filename in os.listdir('xmls'):
    if not filename.endswith('.xml'): continue
    with open(f'xmls/{filename}', 'r') as f:
        soup = BeautifulSoup(f, 'lxml-xml')
    for table in soup.find_all('table'):
        thead = table.find('thead')
        rows = thead.find_all('tr') if thead else table.find_all('tr')[:2]
        headers = []
        for tr in rows:
            for cell in tr.find_all(['th', 'td']):
                headers.append(clean_text(cell.get_text().strip()))
        if any('x' in h for h in headers) or any('coord' in h for h in headers):
            print(f"PMID {filename}: {headers}")
