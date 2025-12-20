import json
from typing import Optional, Dict

import requests

from site_parsers import SiteParserFactory


class NewsParser:
    """Класс для гибридного парсинга новостных сайтов через Oxylabs."""

    def __init__(self, username: str, password: str):
        """
        Инициализация парсера
        
        Args:
            username: Имя пользователя Oxylabs
            password: Пароль Oxylabs
        """
        self.username = username
        self.password = password
        self.api_url = 'https://realtime.oxylabs.io/v1/queries'
    
    def fetch_page_html(self, url: str, retry: int = 2, debug: bool = False) -> Optional[str]:
        """Получение HTML страницы через Oxylabs (без AI парсинга)."""
        payload = {
            'source': 'universal',
            'url': url,
            'parse': False,
            'render': 'html',
            'context': [
                {'key': 'wait_for', 'value': 'networkidle'}
            ]
        }
        
        for attempt in range(retry):
            try:
                response = requests.post(
                    self.api_url,
                    auth=(self.username, self.password),
                    json=payload,
                    timeout=90
                )
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    return data['results'][0].get('content')
                return None
                
            except requests.exceptions.Timeout:
                if attempt < retry - 1:
                    continue
                return None
            except requests.exceptions.RequestException:
                if attempt < retry - 1:
                    continue
                return None
        
        return None
    
    def fetch_and_parse_with_ai(self, url: str, retry: int = 2, debug: bool = False) -> Optional[Dict]:
        """Получение и AI-парсинг новостной статьи через Oxylabs."""
        parsing_instructions = {
            "title": {
                "_fns": [
                    {
                        "_fn": "xpath_one",
                        "_args": ["//h1//text() | //meta[@property='og:title']/@content"]
                    }
                ]
            },
            "text": {
                "_fns": [
                    {
                        "_fn": "xpath",
                        "_args": ["//article//p//text() | //div[contains(@class, 'article')]//p//text() | //div[contains(@class, 'content')]//p//text() | //main//p//text()"]
                    }
                ]
            },
            "author": {
                "_fns": [
                    {
                        "_fn": "xpath_one",
                        "_args": ["//meta[@name='author']/@content | //meta[@property='article:author']/@content | //*[contains(@class, 'author')]//text() | //*[@rel='author']//text()"]
                    }
                ]
            },
            "date": {
                "_fns": [
                    {
                        "_fn": "xpath_one",
                        "_args": ["//time/@datetime | //time//text() | //meta[@property='article:published_time']/@content | //*[contains(@class, 'date')]//text() | //*[contains(@class, 'time')]//text()"]
                    }
                ]
            }
        }
        
        payload = {
            'source': 'universal',
            'url': url,
            'parse': True,
            'parsing_instructions': parsing_instructions,
            'render': 'html',
            'context': [
                {'key': 'wait_for', 'value': 'networkidle'}
            ]
        }
        
        for attempt in range(retry):
            try:
                response = requests.post(
                    self.api_url,
                    auth=(self.username, self.password),
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    if 'content' in result and isinstance(result['content'], dict):
                        return result['content']
                return None

            except requests.exceptions.Timeout:
                if attempt < retry - 1:
                    continue
                return None
            except requests.exceptions.RequestException:
                if attempt < retry - 1:
                    continue
                return None
        
        return None
    
    def process_ai_result(self, ai_data: Dict, debug: bool = False) -> Dict[str, Optional[str]]:
        """Обработка результатов AI-парсинга."""
        result = {
            'title': None,
            'text': None,
            'date': None,
            'author': None
        }
        
        if 'title' in ai_data:
            title = ai_data['title']
            if isinstance(title, list):
                title = title[0] if title else None
            if isinstance(title, str):
                result['title'] = title.strip()
        
        if 'text' in ai_data:
            text_data = ai_data['text']
            if isinstance(text_data, list):
                text_parts = [t.strip() for t in text_data if isinstance(t, str) and len(t.strip()) > 20]
                if text_parts:
                    result['text'] = '\n\n'.join(text_parts)
            elif isinstance(text_data, str):
                result['text'] = text_data.strip()
        
        if 'author' in ai_data:
            author = ai_data['author']
            if isinstance(author, list):
                author = author[0] if author else None
            if isinstance(author, str):
                result['author'] = author.strip()
        
        if 'date' in ai_data:
            date = ai_data['date']
            if isinstance(date, list):
                date = date[0] if date else None
            if isinstance(date, str):
                result['date'] = date.strip()
        
        return result
    
    def get_news_info(self, url: str, debug: bool = False) -> Dict[str, Optional[str]]:
        """Полный процесс парсинга новости (гибридный подход)."""
        parser_class = SiteParserFactory.get_parser(url)
        
        if parser_class:
            html = self.fetch_page_html(url, debug=debug)
            
            if not html:
                return {
                    'title': None,
                    'text': None,
                    'date': None,
                    'author': None,
                    'url': url,
                    'error': 'Не удалось получить страницу'
                }
            
            result = parser_class.parse(html, debug=debug)
            result['url'] = url
            result['parser_type'] = 'specialized'
            
            return result
        else:
            ai_data = self.fetch_and_parse_with_ai(url, debug=debug)
            
            if not ai_data:
                return {
                    'title': None,
                    'text': None,
                    'date': None,
                    'author': None,
                    'url': url,
                    'error': 'Не удалось получить страницу'
                }
            
            result = self.process_ai_result(ai_data, debug=debug)
            result['url'] = url
            result['parser_type'] = 'ai'
            
            return result
