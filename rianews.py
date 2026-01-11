import feedparser
import requests
from bs4 import BeautifulSoup
import time
import spacy

nlp = spacy.load("ru_core_news_sm")

def find_quotes_and_authors(text):
    quotes = []
    in_quote = False
    current = ""
    positions = []
    for i, ch in enumerate(text):
        if ch == '"':
            if in_quote:
                quotes.append(current)
                positions.append(i - len(current) - 1)
                current = ""
                in_quote = False
            else:
                in_quote = True
                current = ""
        elif in_quote:
            current += ch

    results = []
    for quote, pos in zip(quotes, positions):
        context = text[max(0, pos - 60) : pos]
        doc = nlp(context)
        authors = [ent.text for ent in doc.ents if ent.label_ == "PER"]
        results.append({"quote": quote, "authors": authors})

    return results

# Тестовые примеры новостей с цитатами
test_news = [
    {
        "title": "Путин прокомментировал ситуацию на Украине",
        "text": 'Президент России Владимир Путин заявил: "Мы готовы к переговорам, но на наших условиях". Министр иностранных дел Сергей Лавров добавил: "Санкции не сломают нашу экономику".',
        "url": "https://example.com/test1"
    },
    {
        "title": "Экономические реформы в России",
        "text": 'Глава Минфина Антон Силуанов сказал: "Бюджет на следующий год будет сбалансированным". Эксперты отмечают, что "рост ВВП составит 3%".',
        "url": "https://example.com/test2"
    }
]

def test_quotes():
    for news in test_news:
        quotes = find_quotes_and_authors(news["text"])
        print(f"Заголовок: {news['title']}")
        print(f"Текст: {news['text']}")
        print(f"Найдено цитат: {len(quotes)}")
        for quote in quotes:
            print(f"  Цитата: \"{quote['quote']}\"")
            print(f"  Авторы: {quote['authors']}")
        print("-" * 50)

RIA_RSS = "https://ria.ru/export/rss2/archive/index.xml"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def is_target_news(title: str) -> bool:
    title_lower = title.lower()
    return any(k.lower() in title_lower for k in KEYWORDS)


def load_ria_article(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # у РИА основной текст лежит в <div class="article__text">
    article = soup.find("div", class_="article__text")
    if not article:
        return ""

    paragraphs = article.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs)

    return text


def get_ria_news(limit: int = 20):
    feed = feedparser.parse(RIA_RSS)
    print(f"Найдено {len(feed.entries)} записей в RSS")

    news = []

    for entry in feed.entries:
        if len(news) >= limit:
            break

        title = entry.title
        link = entry.link

        try:
            text = load_ria_article(link)
            if text:
                quotes = find_quotes_and_authors(text)
                print(f"В статье '{title}' найдено цитат: {len(quotes)}")
                if quotes:  # Только если есть цитаты
                    news.append({
                        "source": "ria",
                        "title": title,
                        "text": text,
                        "url": link,
                        "quotes": quotes
                    })
                    time.sleep(0.1)  # чтобы не долбить сайт
        except Exception as e:
            print(f"Ошибка при загрузке {link}: {e}")

    return news


if __name__ == "__main__":
    print("Тестирование функции извлечения цитат:")
    test_quotes()
    # print("\nТеперь тестирование с реальными новостями:")
    # news = get_ria_news(1)  # Получить 1 новость с цитатами
    # print(f"Найдено новостей с цитатами: {len(news)}")
    # for item in news:
    #     print(f"Заголовок: {item['title']}")
    #     print(f"URL: {item['url']}")
    #     print("Цитаты:")
    #     for quote in item['quotes']:
    #         print(f"  Цитата: \"{quote['quote']}\"")
    #         print(f"  Авторы: {quote['authors']}")
    #     print("-" * 50)
