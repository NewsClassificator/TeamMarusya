from bs4 import BeautifulSoup
from typing import Optional, Dict
from urllib.parse import urlparse


class SiteParser: # Класс для парсинга конкретного сайта
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(text.split()).strip()
    
    @staticmethod
    def parse(html: str, debug: bool = False) -> Dict[str, Optional[str]]:
        raise NotImplementedError


class LentaParser(SiteParser): 
    @staticmethod
    def parse(html: str, debug: bool = False) -> Dict[str, Optional[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        result = {'title': None, 'text': None, 'date': None, 'author': None}
        
        # Заголовок
        title_tag = soup.find('h1')
        if title_tag:
            result['title'] = SiteParser.clean_text(title_tag.get_text())
            if debug:
                print(f"[Lenta.ru] Заголовок: {result['title'][:50]}...")
        
        # Дата
        time_tag = soup.find('time')
        if time_tag:
            result['date'] = time_tag.get('datetime', time_tag.get_text()).strip()
            if debug:
                print(f"[Lenta.ru] Дата: {result['date']}")
        
        # Автор
        author_elem = soup.find(class_='topic-authors__name')
        if not author_elem:
            author_elem = soup.find(class_='topic-author__name')
        if author_elem:
            author_text = SiteParser.clean_text(author_elem.get_text())
        # Дополнительная информация об авторе
            job_elem = soup.find(class_='topic-authors__job')
            if job_elem:
                job_text = SiteParser.clean_text(job_elem.get_text())
                author_text = f"{author_text} {job_text}"
            result['author'] = author_text
            if debug:
                print(f"[Lenta.ru] Автор: {result['author']}")
        
        # Текст статьи
        article = soup.find('div', class_='topic-body__content')
        if not article:
            article = soup.find('main')
        
        if article:
            # Удаление ненужных элементов
            for tag in article(['script', 'style', 'aside', 'nav', 'header', 'footer']):
                tag.decompose()
            
            paragraphs = article.find_all('p')
            if paragraphs:
                text_parts = [SiteParser.clean_text(p.get_text()) for p in paragraphs 
                             if len(SiteParser.clean_text(p.get_text())) > 30]
                result['text'] = '\n\n'.join(text_parts)
                if debug:
                    print(f"[Lenta.ru] Текст: {len(text_parts)} параграфов, {len(result['text'])} символов")
        
        return result


class RIAParser(SiteParser):
    @staticmethod
    def parse(html: str, debug: bool = False) -> Dict[str, Optional[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        result = {'title': None, 'text': None, 'date': None, 'author': None}
        
        # Заголовок
        title_tag = soup.find('h1', class_='article__title')
        if not title_tag:
            title_tag = soup.find('h1')
        if title_tag:
            result['title'] = SiteParser.clean_text(title_tag.get_text())
            if debug:
                print(f"[RIA.ru] Заголовок: {result['title'][:50]}...")
        
        # Дата
        date_elem = soup.find('div', class_='article__info-date')
        if date_elem:
            time_tag = date_elem.find('a')
            if time_tag:
                result['date'] = SiteParser.clean_text(time_tag.get_text())
                if debug:
                    print(f"[RIA.ru] Дата: {result['date']}")
        
        # Автор
        author_elem = soup.find('div', class_='article__author')
        if author_elem:
            result['author'] = SiteParser.clean_text(author_elem.get_text())
            if debug:
                print(f"[RIA.ru] Автор: {result['author']}")
        
        # Текст - формат RIA.ru
        text_blocks = soup.find_all('div', attrs={'data-type': 'text'})
        if text_blocks:
            text_parts = [SiteParser.clean_text(block.get_text()) for block in text_blocks 
                         if len(SiteParser.clean_text(block.get_text())) > 30]
            result['text'] = '\n\n'.join(text_parts)
            if debug:
                print(f"Текст: {len(text_parts)} блоков")
        else:
            # Запасной вариант
            article = soup.find('div', class_='article__body')
            if article:
                paragraphs = article.find_all('p')
                text_parts = [SiteParser.clean_text(p.get_text()) for p in paragraphs 
                             if len(SiteParser.clean_text(p.get_text())) > 30]
                result['text'] = '\n\n'.join(text_parts)
                if debug:
                    print(f"Параграфов: {len(text_parts)} шт.")
        
        return result


class RBCParser(SiteParser):
    @staticmethod
    def parse(html: str, debug: bool = False) -> Dict[str, Optional[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        result = {'title': None, 'text': None, 'date': None, 'author': None}
        
        # Заголовок
        title_tag = soup.find('h1', class_='article__header__title')
        if not title_tag:
            title_tag = soup.find('h1')
        if title_tag:
            result['title'] = SiteParser.clean_text(title_tag.get_text())
            if debug:
                print(f"[RBC.ru] Заголовок: {result['title'][:50]}...")
        
        # Дата
        time_tag = soup.find('time')
        if time_tag:
            result['date'] = time_tag.get('datetime', time_tag.get_text()).strip()
            if debug:
                print(f"[RBC.ru] Дата: {result['date']}")
        
        # Автор
        author_elem = soup.find('span', class_='article__authors__author')
        if not author_elem:
            author_elem = soup.find('div', class_='article__authors')
        if author_elem:
            result['author'] = SiteParser.clean_text(author_elem.get_text())
            if debug:
                print(f"[RBC.ru] Автор: {result['author']}")
        
        # Текст
        article = soup.find('div', class_='article__text')
        if not article:
            article = soup.find('div', attrs={'itemprop': 'articleBody'})
        
        if article:
            for tag in article(['script', 'style', 'aside', 'nav']):
                tag.decompose()
            
            paragraphs = article.find_all('p')
            text_parts = [SiteParser.clean_text(p.get_text()) for p in paragraphs 
                         if len(SiteParser.clean_text(p.get_text())) > 30]
            result['text'] = '\n\n'.join(text_parts)
            if debug:
                print(f"[RBC.ru] Текст: {len(text_parts)} параграфов")
        
        return result


class RamblerParser(SiteParser):
    @staticmethod
    def parse(html: str, debug: bool = False) -> Dict[str, Optional[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        result = {'title': None, 'text': None, 'date': None, 'author': None}
        
        if debug:
            print("Начало парсинга")
        
        # Заголовок
        title_tag = soup.find('h1', class_='article-header__title')
        if not title_tag:
            title_tag = soup.find('h1')
        if title_tag:
            result['title'] = SiteParser.clean_text(title_tag.get_text())
            if debug:
                print(f"[Rambler.ru] Заголовок: {result['title'][:50]}...")
        
        # Дата
        time_tag = soup.find('time')
        if not time_tag:
            date_elem = soup.find('span', class_='article-header__date')
            if date_elem:
                result['date'] = SiteParser.clean_text(date_elem.get_text())
        else:
            result['date'] = time_tag.get('datetime', time_tag.get_text()).strip()
        
        if result['date'] and debug:
            print(f"[Rambler.ru] Дата: {result['date']}")
        
        # Автор - расширенный поиск
        # Вариант 1: span.article-header__author
        author_elem = soup.find('span', class_='article-header__author')
        
        # Вариант 2: a.autor
        if not author_elem:
            author_elem = soup.find('a', class_='autor')
        
        # Вариант 3: ищем <a> с любыми классами, которые могут содержать имя автора
        if not author_elem:
            # Ищем все ссылки в header или в первой части статьи
            header = soup.find('header') or soup.find('div', class_='article-header')
            if header:
                for link in header.find_all('a'):
                    # Проверяем, содержит ли ссылка имя (обычно начинается с заглавной буквы)
                    text = SiteParser.clean_text(link.get_text())
                    if text and len(text) < 50 and text[0].isupper():
                        # Проверяем классы на наличие паттернов типа "udVRHbll M5Yse4nk"
                        classes = link.get('class', [])
                        if classes and len(text.split()) <= 3:  # Имя обычно 1-3 слова
                            author_elem = link
                            if debug:
                                print(f"[Rambler.ru] Найден автор через heuristic search: {text}, classes: {classes}")
                            break
        
        if author_elem:
            result['author'] = SiteParser.clean_text(author_elem.get_text())
            if debug:
                print(f"[Rambler.ru] Автор: {result['author']}")
        elif debug:
            print("[Rambler.ru] Автор не найден")
        
        # Текст
        article = soup.find('div', class_='article-body')
        if not article:
            article = soup.find('article')
        
        if article:
            for tag in article(['script', 'style', 'aside', 'nav']):
                tag.decompose()
            
            paragraphs = article.find_all('p')
            text_parts = [SiteParser.clean_text(p.get_text()) for p in paragraphs 
                         if len(SiteParser.clean_text(p.get_text())) > 30]
            result['text'] = '\n\n'.join(text_parts)
            if debug:
                print(f"[Rambler.ru] Текст найден: {len(text_parts)} параграфов")
        
        if debug:
            print(f"[Rambler.ru] Результат: title={bool(result['title'])}, text={bool(result['text'])}, date={bool(result['date'])}, author={bool(result['author'])}")
        
        return result


class MailNewsParser(SiteParser):
    @staticmethod
    def parse(html: str, debug: bool = False) -> Dict[str, Optional[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        result = {'title': None, 'text': None, 'date': None, 'author': None}
        
        if debug:
            print("[Mail.ru News] Начало парсинга")
        
        # Заголовок
        title_tag = soup.find('h1', class_='hdr__inner')
        if not title_tag:
            title_tag = soup.find('h1')
        if title_tag:
            result['title'] = SiteParser.clean_text(title_tag.get_text())
            if debug:
                print(f"[Mail.ru News] Заголовок найден: {result['title'][:50]}...")
        
        # Дата
        time_tag = soup.find('time')
        if not time_tag:
            date_elem = soup.find('span', class_='note__date')
            if date_elem:
                result['date'] = SiteParser.clean_text(date_elem.get_text())
        else:
            result['date'] = time_tag.get('datetime', time_tag.get_text()).strip()
        
        if result['date'] and debug:
            print(f"[Mail.ru News] Дата найдена: {result['date']}")
        
        # Автор - также проверяем meta тег
        author_elem = soup.find('meta', attrs={'name': 'author'})
        if author_elem:
            result['author'] = author_elem.get('content', '').strip()
            if debug:
                print(f"[Mail.ru News] Автор из meta: {result['author']}")
        else:
            author_elem = soup.find('span', class_='note__author')
            if not author_elem:
                author_elem = soup.find('a', class_='link link_muted')
            if author_elem:
                result['author'] = SiteParser.clean_text(author_elem.get_text())
                if debug:
                    print(f"[Mail.ru News] Автор найден: {result['author']}")
        
        # Текст - блоки article-item-type="html"
        article_blocks = soup.find_all('div', attrs={'article-item-type': 'html'})
        if article_blocks:
            if debug:
                print(f"[Mail.ru News] Найдено {len(article_blocks)} блоков article-item-type='html'")
            text_parts = []
            for block in article_blocks:
                paragraphs = block.find_all('p')
                for p in paragraphs:
                    cleaned_text = SiteParser.clean_text(p.get_text())
                    if len(cleaned_text) > 30:
                        text_parts.append(cleaned_text)
            
            if text_parts:
                result['text'] = '\n\n'.join(text_parts)
                if debug:
                    print(f"[Mail.ru News] Текст найден: {len(text_parts)} параграфов, {len(result['text'])} символов")
        
        # Запасной вариант для старого формата
        if not result['text']:
            article = soup.find('div', class_='article__item_alignment_left')
            if not article:
                article = soup.find('div', class_='article__body')
            if not article:
                article = soup.find('div', attrs={'itemprop': 'articleBody'})
            if not article:
                article = soup.find('article')
            
            if article:
                for tag in article(['script', 'style', 'aside', 'nav', 'figure']):
                    tag.decompose()
                
                # Удаляем рекламу и подписки
                for ad in article.find_all(class_=lambda x: x and any(word in x.lower() for word in ['banner', 'promo', 'subscribe'])):
                    ad.decompose()
                
                paragraphs = article.find_all('p')
                text_parts = [SiteParser.clean_text(p.get_text()) for p in paragraphs 
                             if len(SiteParser.clean_text(p.get_text())) > 30]
                if text_parts:
                    result['text'] = '\n\n'.join(text_parts)
                    if debug:
                        print(f"[Mail.ru News]{len(text_parts)} параграфов")
        
        if debug:
            print(f"[Mail.ru News] Результат: title={bool(result['title'])}, text={bool(result['text'])}, date={bool(result['date'])}, author={bool(result['author'])}")
        
        return result


class SiteParserFactory:
    """Фабрика для выбора правильного парсера на основе домена"""
    
    SUPPORTED_SITES = {
        'lenta.ru': LentaParser,
        'ria.ru': RIAParser,
        'rbc.ru': RBCParser,
        'rambler.ru': RamblerParser,
        'news.mail.ru': MailNewsParser,
    }
    
    @classmethod
    def get_parser(cls, url: str) -> Optional[type]:
        """
        Получить парсер для URL
        
        Args:
            url: URL новостной статьи
            
        Returns:
            Класс парсера или None если сайт не поддерживается
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Убираем www. если есть
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Проверяем прямое совпадение
            parser = cls.SUPPORTED_SITES.get(domain)
            if parser:
                return parser
            
            # Проверяем поддомены Rambler (news.rambler.ru, finance.rambler.ru, sport.rambler.ru и т.д.)
            if domain.endswith('.rambler.ru'):
                return RamblerParser
            
            return None
        except Exception:
            return None
    
    @classmethod
    def is_supported(cls, url: str) -> bool:
        """Проверить, поддерживается ли сайт"""
        return cls.get_parser(url) is not None
    
    @classmethod
    def get_supported_sites(cls) -> list:
        """Получить список поддерживаемых сайтов"""
        return list(cls.SUPPORTED_SITES.keys())
