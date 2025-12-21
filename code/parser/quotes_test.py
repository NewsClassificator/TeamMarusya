def _extract_quotes(text: str):
    """
    Вспомогательная функция: находит все фрагменты в кавычках.

    Returns:
        quotes: список строк без кавычек
        spans: список кортежей (start, end), индексы в исходном тексте
               start/end включают сами кавычки.
    """
    quotes = []
    spans = []
    current = []
    start_idx = None
    expected_closer = None

    pairs = {
        '"': '"',
        "«": "»",
        "“": "”",
        "„": "“",
    }
    openers = set(pairs.keys())
    closers = set(pairs.values())

    for i, ch in enumerate(text):
        if expected_closer is None and ch in openers:
            # start quote
            start_idx = i
            expected_closer = pairs[ch]
            current = []
        elif expected_closer is not None and ch == expected_closer:
            quotes.append("".join(current))
            spans.append((start_idx, i))
            current = []
            start_idx = None
            expected_closer = None
        elif expected_closer is not None:
            current.append(ch)
        else:
            # handle symmetric closers acting as openers (e.g., stray closing quote)
            if ch in closers and ch in openers:
                start_idx = i
                expected_closer = pairs.get(ch, ch)
                current = []

    return quotes, spans


def _guess_authors(context: str) -> list[str]:
    """
    Простая эвристика для поиска автора: последняя последовательность слов с прописной буквы
    перед цитатой. Возвращает список потенциальных авторов (может быть пустым).
    """
    import re

    candidates = re.findall(r"([А-ЯЁ][а-яё]+(?:\\s+[А-ЯЁ][а-яё]+){0,2})", context)
    if not candidates:
        return []
    return [candidates[-1].strip()]


def find_quotes_and_authors(text: str):
    """
    Находит цитаты и возможных авторов рядом с ними (по контексту слева).
    """
    quotes, spans = _extract_quotes(text)
    results = []
    for quote, (start, _end) in zip(quotes, spans):
        context = text[max(0, start - 80) : start]
        authors = _guess_authors(context)
        results.append({"quote": quote, "authors": authors})

    return results


def replace_quotes_with_placeholder(text: str, placeholder: str = "цитата") -> str:
    """
    Возвращает текст, в котором все фразы в кавычках заменены на слово placeholder.

    Пример:
        'Он сказал: "Привет мир"' -> 'Он сказал: цитата'
    """
    _, spans = _extract_quotes(text)
    if not spans:
        return text

    parts = []
    last_idx = 0
    for start, end in spans:
        parts.append(text[last_idx:start])
        parts.append(placeholder)
        last_idx = end + 1

    parts.append(text[last_idx:])
    return "".join(parts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        print("Введите текст для анализа (Ctrl+D для завершения):")
        input_text = sys.stdin.read().strip()

    if not input_text:
        print("Пустой текст, ничего анализировать.")
        sys.exit(0)

    print("=== ЦИТАТЫ И АВТОРЫ ===")
    for idx, item in enumerate(find_quotes_and_authors(input_text), start=1):
        print(f"{idx}. \"{item['quote']}\"")
        if item["authors"]:
            print(f"   Возможные авторы: {', '.join(item['authors'])}")
        else:
            print("   Авторы не найдены")

    print("\n=== ТЕКСТ БЕЗ ЦИТАТ ===")
    print(replace_quotes_with_placeholder(input_text))
