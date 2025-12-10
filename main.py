"""
Гибридный парсер новостных сайтов с использованием Oxylabs
Использует специализированные парсеры для популярных сайтов и AI для остальных
"""
import requests
from pprint import pprint
from typing import Optional, Dict
import json
from site_parsers import SiteParserFactory


class NewsParser:
    """Класс для гибридного парсинга новостных сайтов через Oxylabs"""
    
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
        """
        Получение HTML страницы через Oxylabs (без AI парсинга)
        
        Args:
            url: URL новостной статьи
            retry: Количество повторных попыток при ошибке
            debug: Включить отладочную информацию
            
        Returns:
            HTML содержимое страницы или None в случае ошибки
        """
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
                if debug:
                    print(f"[DEBUG] Получение HTML, попытка {attempt + 1}/{retry}...")
                else:
                    print(f"⚡ Загрузка страницы, попытка {attempt + 1}/{retry}...")
                    
                response = requests.post(
                    self.api_url,
                    auth=(self.username, self.password),
                    json=payload,
                    timeout=90
                )
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    html = data['results'][0].get('content')
                    if debug:
                        print(f"[DEBUG] Получен HTML размером: {len(html)} символов")
                    return html
                return None
                
            except requests.exceptions.Timeout:
                if attempt < retry - 1:
                    print(f"⏱️  Timeout - пробуем еще раз...")
                    continue
                else:
                    print(f"❌ Ошибка: превышено время ожидания после {retry} попыток")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"❌ Ошибка при запросе: {e}")
                if attempt < retry - 1:
                    print(f"🔄 Пробуем еще раз...")
                    continue
                return None
        
        return None
    
    def fetch_and_parse_with_ai(self, url: str, retry: int = 2, debug: bool = False) -> Optional[Dict]:
        """
        Получение и AI-парсинг новостной статьи через Oxylabs
        
        Args:
            url: URL новостной статьи
            retry: Количество повторных попыток при ошибке
            debug: Включить отладочную информацию
            
        Returns:
            Словарь с данными новости или None в случае ошибки
        """
        # Схема данных для AI парсера
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
                if debug:
                    print(f"[DEBUG] Отправка AI-запроса, попытка {attempt + 1}/{retry}...")
                else:
                    print(f"🤖 AI парсинг, попытка {attempt + 1}/{retry}...")
                    
                response = requests.post(
                    self.api_url,
                    auth=(self.username, self.password),
                    json=payload,
                    timeout=120  # AI парсинг может занять больше времени
                )
                response.raise_for_status()
                
                data = response.json()
                
                if debug:
                    print(f"[DEBUG] Получен ответ от Oxylabs")
                    print(f"[DEBUG] Структура ответа: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}...")
                
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    
                    # Извлекаем спарсенные данные
                    if 'content' in result and isinstance(result['content'], dict):
                        parsed_data = result['content']
                        if debug:
                            print(f"[DEBUG] AI нашел данные: {list(parsed_data.keys())}")
                        return parsed_data
                    
                return None
                
            except requests.exceptions.Timeout:
                if attempt < retry - 1:
                    print(f"⏱️  Timeout - пробуем еще раз...")
                    continue
                else:
                    print(f"❌ Ошибка: превышено время ожидания после {retry} попыток")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"❌ Ошибка при запросе: {e}")
                if attempt < retry - 1:
                    print(f"🔄 Пробуем еще раз...")
                    continue
                return None
        
        return None
    
    def process_ai_result(self, ai_data: Dict, debug: bool = False) -> Dict[str, Optional[str]]:
        """
        Обработка результатов AI-парсинга
        
        Args:
            ai_data: Данные от AI парсера
            debug: Включить отладочную информацию
            
        Returns:
            Словарь с очищенными данными новости
        """
        result = {
            'title': None,
            'text': None,
            'date': None,
            'author': None
        }
        
        # Обработка заголовка
        if 'title' in ai_data:
            title = ai_data['title']
            if isinstance(title, list):
                title = title[0] if title else None
            if isinstance(title, str):
                result['title'] = title.strip()
                if debug:
                    print(f"[DEBUG] Заголовок: {result['title'][:100]}...")
        
        # Обработка текста
        if 'text' in ai_data:
            text_data = ai_data['text']
            if isinstance(text_data, list):
                # Фильтруем короткие фрагменты (меньше 20 символов)
                text_parts = [t.strip() for t in text_data if isinstance(t, str) and len(t.strip()) > 20]
                if text_parts:
                    result['text'] = '\n\n'.join(text_parts)
                    if debug:
                        print(f"[DEBUG] Текст: найдено {len(text_parts)} параграфов, общая длина: {len(result['text'])} символов")
            elif isinstance(text_data, str):
                result['text'] = text_data.strip()
                if debug:
                    print(f"[DEBUG] Текст: {len(result['text'])} символов")
        
        # Обработка автора
        if 'author' in ai_data:
            author = ai_data['author']
            if isinstance(author, list):
                author = author[0] if author else None
            if isinstance(author, str):
                result['author'] = author.strip()
                if debug:
                    print(f"[DEBUG] Автор: {result['author']}")
        
        # Обработка даты
        if 'date' in ai_data:
            date = ai_data['date']
            if isinstance(date, list):
                date = date[0] if date else None
            if isinstance(date, str):
                result['date'] = date.strip()
                if debug:
                    print(f"[DEBUG] Дата: {result['date']}")
        
        return result
    
    def get_news_info(self, url: str, debug: bool = False) -> Dict[str, Optional[str]]:
        """
        Полный процесс парсинга новости (гибридный подход)
        
        Args:
            url: URL новостной статьи
            debug: Включить отладочную информацию
            
        Returns:
            Словарь с данными новости
        """
        print(f"📡 Получаем и парсим страницу: {url}")
        
        # Проверяем, есть ли специализированный парсер для этого сайта
        parser_class = SiteParserFactory.get_parser(url)
        
        if parser_class:
            # Используем быстрый специализированный парсер
            site_name = parser_class.__name__.replace('Parser', '')
            print(f"⚡ Используется специализированный парсер для {site_name}")
            
            html = self.fetch_page_html(url, debug=debug)
            
            if not html:
                print("❌ Не удалось получить страницу")
                return {
                    'title': None,
                    'text': None,
                    'date': None,
                    'author': None,
                    'url': url,
                    'error': 'Не удалось получить страницу'
                }
            
            print("✅ Парсим данные...")
            result = parser_class.parse(html, debug=debug)
            result['url'] = url
            result['parser_type'] = 'specialized'
            
            return result
        else:
            # Используем AI-парсер для неизвестных сайтов
            print(f"🤖 Сайт не распознан, используется AI-парсинг от Oxylabs...")
            
            ai_data = self.fetch_and_parse_with_ai(url, debug=debug)
            
            if not ai_data:
                print("❌ Не удалось получить или спарсить страницу")
                return {
                    'title': None,
                    'text': None,
                    'date': None,
                    'author': None,
                    'url': url,
                    'error': 'Не удалось получить страницу'
                }
            
            print("✅ Данные получены, обрабатываем...")
            result = self.process_ai_result(ai_data, debug=debug)
            result['url'] = url
            result['parser_type'] = 'ai'
            
            return result


def main():
    """Основная функция"""
    # Ваши учетные данные Oxylabs
    USERNAME = 'termenater_fviy2'
    PASSWORD = 'NAY9+Beh9S=WhtX'
    
    # Создаем парсер
    parser = NewsParser(USERNAME, PASSWORD)
    
    # Запрашиваем URL у пользователя
    print("=" * 80)
    print("🚀 Гибридный парсер новостных сайтов")
    print("=" * 80)
    print("\n⚡ Быстрый парсинг для популярных сайтов:")
    for site in SiteParserFactory.get_supported_sites():
        print(f"   • {site}")
    print("\n🤖 AI-парсинг для всех остальных сайтов!")
    
    url = input("\n🔗 Введите URL новостной статьи: ").strip()
    
    if not url:
        print("❌ Ошибка: URL не может быть пустым")
        return
    
    # Проверяем, что это похоже на URL
    if not url.startswith(('http://', 'https://')):
        print("❌ Ошибка: URL должен начинаться с http:// или https://")
        return
    
    # Спрашиваем о режиме отладки
    debug_mode = input("🔍 Включить режим отладки? (y/n, по умолчанию n): ").strip().lower() == 'y'
    
    # Получаем информацию
    print("\n" + "=" * 80)
    news_info = parser.get_news_info(url, debug=debug_mode)
    
    # Выводим результаты
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ПАРСИНГА")
    print("=" * 80)
    
    print(f"\n📰 ЗАГОЛОВОК:")
    print(f"   {news_info.get('title', 'Не найдено')}")
    
    print(f"\n✍️  АВТОР:")
    print(f"   {news_info.get('author', 'Не найдено')}")
    
    print(f"\n📅 ДАТА ПУБЛИКАЦИИ:")
    print(f"   {news_info.get('date', 'Не найдено')}")
    
    print(f"\n📝 ТЕКСТ СТАТЬИ:")
    text = news_info.get('text', 'Не найдено')
    if text and len(text) > 500:
        print(f"   {text[:500]}...")
        print(f"\n   (Показано первые 500 символов из {len(text)})")
    else:
        print(f"   {text}")
    
    print("\n" + "=" * 80)
    
    # Для отладки - выводим полный JSON
    print("\nПолные данные в JSON формате:")
    print("-" * 80)
    pprint(news_info)


if __name__ == "__main__":
    main()
