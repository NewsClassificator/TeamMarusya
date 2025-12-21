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