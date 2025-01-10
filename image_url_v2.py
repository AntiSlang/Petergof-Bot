import json
import os
from logging import Handler, LogRecord
import re
from pathlib import Path
from icrawler.builtin import GoogleImageCrawler
import requests


def load_dictionary(path='users.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_dictionary(dictionary, path='users.json'):
    Path(path).write_text(json.dumps(dictionary, ensure_ascii=False, sort_keys=False, indent=4), encoding='utf-8')


def valid(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        return content_type.startswith('image/')
    except requests.RequestException:
        return False


class URLCollectorHandler(Handler):
    def __init__(self):
        super().__init__()
        self.collected_urls = []
    def emit(self, record: LogRecord):
        match = re.search(r"image #\d+\t(.*)", record.getMessage())
        if match:
            self.collected_urls.append(match.group(1))


def fetch_image_url(query):
    url_handler = URLCollectorHandler()
    crawler = GoogleImageCrawler()
    crawler.downloader.logger.addHandler(url_handler)
    crawler.crawl(keyword=query, max_num=1)
    return url_handler.collected_urls[0] if url_handler.collected_urls else None


data = load_dictionary('data.json')
for i in data['places']:
    if valid(i['image_url']):
        i['image_url_v2'] = i['image_url']
    else:
        i['image_url_v2'] = fetch_image_url(i['title'])
        os.remove('images/000001.jpg')
write_dictionary(data, 'data.json')
