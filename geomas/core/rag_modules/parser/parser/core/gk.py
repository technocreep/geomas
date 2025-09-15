import re
from time import sleep
import aiohttp
import asyncio
import os
import json
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from parser.core.utils.logger import Logger
from pathlib import Path
from tqdm import tqdm



class GKParser:
    """Parse files from GeoKniga.org"""
    def __init__(
            self,
            max_pages: int = 10,
            file_limit: int = 10,
            ):
        self.max_pages = max_pages
        self.file_limit = file_limit
        self.SEM_LIMIT = 3 # simultaneous downloads
        self.BASE_URL = "https://www.geokniga.org"
        self.logger = Logger.get()
        
        # if debug
        # self.logger = Logger.create(source='geokniga')
        self.results_dir = self.logger.results_dir

    def run(self,):
        self.logger.info('Process started')
        books_path = self.search_books()
        self.load_books_and_download(path=books_path)
        self.logger.info('Process finished')



    def search_books(self):
        results = []
        for page in range(0, self.max_pages):
            self.logger.info(f'Searching for books on {page+1}/{self.max_pages} page...')
            params = {'field_temat': 1}  # examples for "геологоразведка" topic
            
            URL = f'https://www.geokniga.org/books?page={page}&field_title=&field_author=&field-redaktor=&field_temat={params["field_temat"]}&field_labels=&field_izdat=&field-lang%5B%5D=1292&field-lang%5B%5D=3048&field-lang%5B%5D=50884&field-lang%5B%5D=44436&field-lang%5B%5D=47185&field-lang%5B%5D=31179&field-lang%5B%5D=42324&field-lang%5B%5D=4591&field-lang%5B%5D=55125&field-lang%5B%5D=6877&field-lang%5B%5D=53343&field-lang%5B%5D=67042&field-lang%5B%5D=4574&field-lang%5B%5D=6467&field-lang%5B%5D=18864'
            resp = requests.get(URL)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            self.logger.info(f"Found {len(soup.select('div.book_body_title'))} book links")
            
            for idx, div in enumerate(soup.select('div.book_body_title')):
                link_tag = div.find('a')
                if link_tag:
                    title = link_tag.text.strip()
                    relative_url = link_tag['href']
                    full_url = urljoin(self.BASE_URL, relative_url)

                    author = soup.select('div.book_body_author')[idx].text.split('Автор(ы):')[-1].strip()
                    izdat_text = soup.select('div.book_body_izdat_full')[idx].text
                    match = re.search(r'(\d{4})\s*г\.', izdat_text)
                    # if match:
                    year = int(match.group(1))

                    annot = soup.select('div.book_body_annot')[idx].text


                    results.append({
                        'title': title, 
                        'year': year,
                        'author': author,
                        'url': full_url,
                        'abstract': annot
                        })
            sleep(2)
        
        self.save_results_to_json(
            results=results,
            filename=self.results_dir / 'geosearch.json'
        )
        self.logger.info(f'List of books is saved at:')
        self.logger.info(self.results_dir / 'geosearch.json')

        return os.path.abspath(self.results_dir / 'geosearch.json')

    def load_books_and_download(self, path):
        with open(path, 'r') as f:
            books = json.load(f)
        asyncio.run(self.download_books_async(books))

    async def download_books_async(self, book_list):
        self.logger.info(f'Loading {self.file_limit}/{len(book_list)} books')
        if self.file_limit == 0:
            return 
        sem = asyncio.Semaphore(self.SEM_LIMIT)
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.process_book(sem, session, book)
                for book in book_list[:self.file_limit]
            ]
            await asyncio.gather(*tasks)

    async def process_book(self, sem, session, book):
        async with sem:
            try:
                down_url = await self.get_download_link(session, book['url'])
                if down_url:
                    await asyncio.sleep(random.uniform(1.5, 4.0))  # задержка перед скачиванием
                    await self.download_pdf(
                        session, 
                        down_url, 
                        dest_folder=self.results_dir / 'downloads'
                        )
            except Exception as e:
                self.logger.error(f"[!] Error during processing {book['title']}: {e}")

    async def fetch(self, session, url):
        async with session.get(url) as resp:
            return await resp.text()

    async def get_download_link(self, session, book_page_url):
        """Finds link to download file. Preferably in pdf format"""
        html = await self.fetch(session, book_page_url)
        soup = BeautifulSoup(html, 'html.parser')
        link = soup.select_one('a[href$=".pdf"]') or soup.select_one('a[href$=".djvu"]')
        return urljoin(self.BASE_URL, link['href']) if link else None

    async def download_pdf(self, session, url, dest_folder='downloads'):
        os.makedirs(dest_folder, exist_ok=True)
        filename = os.path.join(dest_folder, os.path.basename(url))
        async with session.get(url) as resp:
            with open(filename, 'wb') as f:
                while True:
                    chunk = await resp.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
        self.logger.info(f"[✔] Downloaded: {filename}")

    def save_results_to_json(self, results, filename='books.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    gk = GKParser(
        max_pages=1,
        file_limit=2
    )
    gk.run()

