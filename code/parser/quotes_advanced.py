"""
Advanced quote extraction using Natasha NLP library
Based on the implementation from quote_extractor.ipynb
"""

from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc

segmenter = Segmenter()
emb = NewsEmbedding()
ner = NewsNERTagger(emb)

SPEECH_VERB_ROOTS = [
    'сказ', 'заяв', 'отмет', 'подчеркн', 'сообщ',
    'добав', 'поясн', 'рассказ', 'указ', 'коммент', 'пис'
]

ROLES = {
    'министр финансов',
    'президент', 'министр', 'губернатор',
    'депутат', 'премьер', 'глава', 'директор',
    'артист', 'лауреат',
    'профессор', 'академик', 'редактор', 'журналист',
    'пресс-секретарь',
    'спикер', 'сенатор', 'конгрессмен', 'мэр',
    'генерал', 'полковник', 'капитан',
    'актер', 'актриса', 'режиссер', 'продюсер',
    'писатель', 'поэт', 'художник',
    'ученый', 'исследователь', 'инженер',
}


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


def extract_filtered_quotes(text, context_chars=50, min_words=3):
    """
    Advanced quote extraction with author identification using NER and linguistic rules

    Args:
        text: Input text to extract quotes from
        context_chars: Number of context characters to consider around quotes
        min_words: Minimum number of words for a quote to be considered valid

    Returns:
        List of dictionaries with 'quote' and 'author' keys
    """
    import re
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()

    quotes, spans = _extract_quotes(text)

    if not quotes:
        return []

    # 2. NER
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner)
    persons = [s for s in doc.spans if s.type == 'PER']

    results = []

    for quote, (qs, qe) in zip(quotes, spans):

        is_short = len(quote.split()) < min_words

        # 3. Контекст
        left_idx = max(0, qs - context_chars)
        right_idx = min(len(text), qe + context_chars)
        context = text[left_idx:right_idx].lower()

        # 4. Проверка глагола речи
        if not any(root in context for root in SPEECH_VERB_ROOTS):
            continue

        author = None

        # 5. Поиск PER слева (без глагола речи между)
        left_per_candidates = []
        for p in persons:
            if p.stop <= qs and qs - p.stop < 150:
                between = text[p.stop:qs].lower()
                # если между PER и цитатой есть глагол речи — PER не автор
                if any(root in between for root in SPEECH_VERB_ROOTS):
                    continue
                left_per_candidates.append(p)

        if left_per_candidates:
            p = min(left_per_candidates, key=lambda p: qs - p.stop)
            author = text[p.start:p.stop]

        # 6. Если PER не найден слева — ищем справа (для конструкций типа "...цитата...", — сказал X)
        if not author:
            right_per_candidates = []
            for p in persons:
                if p.start >= qe and p.start - qe < 150:  # PER справа от цитаты
                    between = text[qe:p.start].lower()
                    # проверяем, есть ли глагол речи между цитатой и PER
                    if any(root in between for root in SPEECH_VERB_ROOTS):
                        right_per_candidates.append(p)

            if right_per_candidates:
                p = min(right_per_candidates, key=lambda p: p.start - qe)
                author = text[p.start:p.stop]

        # 7. Если PER не найден — ищем роль слева
        if not author:
            closest_verb_idx = -1

            for root in SPEECH_VERB_ROOTS:
                idx = text[:qs].rfind(root)
                if idx > closest_verb_idx:
                    closest_verb_idx = idx

            if closest_verb_idx != -1:
                sentence_start = max(
                    text[:closest_verb_idx].rfind('.'),
                    text[:closest_verb_idx].rfind('!'),
                    text[:closest_verb_idx].rfind('?'),
                    text[:closest_verb_idx].rfind(':')
                )

                snippet = text[sentence_start + 1:closest_verb_idx].lower()

                matches = [role for role in ROLES if role in snippet]
                if matches:
                    author = max(matches, key=len).capitalize()

        # 8. Если всё ещё нет автора — ищем роль справа (после цитаты)
        if not author:
            # Ищем глагол речи справа от цитаты
            closest_right_verb_idx = -1
            for root in SPEECH_VERB_ROOTS:
                idx = text[qe:].find(root)
                if idx != -1:
                    actual_idx = qe + idx
                    if closest_right_verb_idx == -1 or actual_idx < closest_right_verb_idx:
                        closest_right_verb_idx = actual_idx

            if closest_right_verb_idx != -1:
                # Ищем конец предложения после глагола речи
                sentence_end = min(
                    text[closest_right_verb_idx:].find('.'),
                    text[closest_right_verb_idx:].find('!'),
                    text[closest_right_verb_idx:].find('?'),
                    text[closest_right_verb_idx:].find(',')
                )

                if sentence_end == -1:  # Если не найдено стандартных знаков препинания
                    sentence_end = len(text) - closest_right_verb_idx

                if sentence_end > 0:
                    snippet = text[closest_right_verb_idx:closest_right_verb_idx + sentence_end].lower()

                    matches = [role for role in ROLES if role in snippet]
                    if matches:
                        author = max(matches, key=len).capitalize()

        # 9. Если всё ещё нет автора, пробуем найти существительное в именительном падеже в пределах одного предложения с цитатой
        if not author:
            # Найдем границы предложения, содержащего цитату
            # Ищем начало предложения (после точки или начало текста)
            dot_pos = text.rfind('.', 0, qs)
            excl_pos = text.rfind('!', 0, qs)
            quest_pos = text.rfind('?', 0, qs)

            # Находим максимальную позицию среди всех знаков препинания
            start_sentence = max(-1, dot_pos, excl_pos, quest_pos) + 1  # +1 чтобы начать после знака препинания

            # Ищем конец предложения (до точки или конец текста)
            dot_end = text.find('.', qe)
            excl_end = text.find('!', qe)
            quest_end = text.find('?', qe)

            # Находим минимальную позицию среди всех знаков препинания после цитаты
            ends = [x for x in [dot_end, excl_end, quest_end] if x != -1]
            if ends:
                end_sentence = min(ends)
            else:
                end_sentence = len(text)  # если нет знаков препинания, до конца текста

            sentence_with_quote = text[start_sentence:end_sentence].strip()

            # Ищем глагол речи в пределах этого предложения
            closest_verb_idx_in_sentence = -1
            for root in SPEECH_VERB_ROOTS:
                idx = sentence_with_quote.lower().find(root)
                if idx != -1 and (closest_verb_idx_in_sentence == -1 or idx < closest_verb_idx_in_sentence):
                    closest_verb_idx_in_sentence = idx

            if closest_verb_idx_in_sentence != -1:
                # Преобразуем индекс в пределах предложения к абсолютному индексу в тексте
                closest_verb_idx = start_sentence + closest_verb_idx_in_sentence

                # Ищем существительное в именительном падеже рядом с глаголом в пределах этого же предложения
                # Сначала ищем влево от глагола (в пределах предложения)
                search_start = max(start_sentence, closest_verb_idx - 50)
                search_text_before = text[search_start:closest_verb_idx]

                # Ищем слова в именительном падеже (существительные, которые могут быть профессией/ролью)
                words_before = re.findall(r'[а-яёА-ЯЁ]+', search_text_before)
                for word in reversed(words_before):  # идём с конца, чтобы найти ближайшее к глаголу
                    if len(word) < 3:  # пропускаем короткие слова
                        continue
                    parsed_word = morph.parse(word)[0] if morph.parse(word) else None
                    if parsed_word and 'NOUN' in parsed_word.tag and 'nomn' in parsed_word.tag:
                        # Проверяем, является ли это слово профессией/ролью из нашего списка
                        word_lower = word.lower()
                        if any(word_lower in role.lower() for role in ROLES) or word_lower in [r.split()[-1].lower() for r in ROLES]:
                            # Это известная роль, используем её
                            # Пытаемся найти полную фразу с профессией
                            search_range = text[max(start_sentence, closest_verb_idx-50):closest_verb_idx]

                            # Ищем полную фразу, например "сварщик, допрошенный корреспондентами"
                            pattern = r'([а-яёА-ЯЁ]+(?:\s*,\s*[а-яёА-ЯЁ]+(?:\s+[а-яёА-ЯЁ]+)*)*\s+' + re.escape(word) + r'|' + re.escape(word) + r')(?:\s*,\s*|\s+|$|[.!?;:])'
                            matches = re.findall(pattern, search_range)
                            if matches:
                                # Берем последнее совпадение (ближайшее к глаголу)
                                full_role = matches[-1].strip()
                                author = full_role.capitalize()
                            else:
                                author = word.capitalize()
                            break
                        else:
                            # Это существительное в именительном падеже, но неизвестная роль
                            # Проверим, может ли оно быть профессией/ролью (например, "сыщик", "ученый", "артист")
                            # Если слово выглядит как профессия (обычно короче 15 символов и не является обычным существительным), используем его
                            if len(word) <= 15 and word_lower not in ['год', 'день', 'время', 'дело', 'жизнь', 'рука', 'лицо', 'глаз', 'страна', 'мир', 'слово', 'работа', 'голова', 'часть', 'место', 'лицо']:
                                # Пытаемся найти полную фразу
                                search_range = text[max(start_sentence, closest_verb_idx-50):closest_verb_idx]
                                pattern = r'([а-яёА-ЯЁ]+(?:\s*,\s*[а-яёА-ЯЁ]+(?:\s+[а-яёА-ЯЁ]+)*)*\s+' + re.escape(word) + r'|' + re.escape(word) + r')(?:\s*,\s*|\s+|$|[.!?;:])'
                                matches = re.findall(pattern, search_range)
                                if matches:
                                    # Берем последнее совпадение (ближайшее к глаголу)
                                    full_role = matches[-1].strip()
                                    author = full_role.capitalize()
                                else:
                                    author = word.capitalize()
                                break

                # Если не нашли до глагола, ищем после (в пределах предложения)
                if not author:
                    search_end = min(end_sentence, closest_verb_idx + 50)
                    search_text_after = text[closest_verb_idx:search_end]

                    # Ищем слова после глагола речи
                    words_after = re.findall(r'[а-яёА-ЯЁ]+', search_text_after)
                    for word in words_after:
                        if len(word) < 3:  # слишком короткое слово
                            continue
                        parsed_word = morph.parse(word)[0] if morph.parse(word) else None
                        if parsed_word and 'NOUN' in parsed_word.tag and 'nomn' in parsed_word.tag:
                            # Проверяем, является ли это слово профессией/ролью из нашего списка
                            word_lower = word.lower()
                            if any(word_lower in role.lower() for role in ROLES) or word_lower in [r.split()[-1].lower() for r in ROLES]:
                                author = word.capitalize()
                                break

        # 10. Отсев мусорных коротких заголовков
        if quote.istitle() and len(quote.split()) <= 4:
            continue

        # Если цитата содержит глаголы речи, но нет автора, всё равно включаем её (но с предупреждением)
        # Но не включаем слишком короткие цитаты без автора
        if is_short and not author:
            continue

        results.append({
            "quote": quote,
            "authors": [author] if author else []  # Возвращаем цитату даже без автора
        })

    return results


def find_quotes_and_authors(text):
    """
    Wrapper function to maintain compatibility with existing interface
    """
    return extract_filtered_quotes(text)


def replace_quotes_with_placeholder(text, placeholder="цитата"):
    """
    Replace quoted phrases with a placeholder, maintaining compatibility with existing interface
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
