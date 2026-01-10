import json
import os
import requests
import time

API_KEY = os.getenv("GPUSTACK_API_KEY")
if not API_KEY:
    raise RuntimeError("GPUSTACK_API_KEY not set")

URL = "https://gpustack.gojdaforever.ru:1443/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

MODEL = "gemma-3-12b-awq"


def label_text(text: str) -> str:
    prompt = f"""
Определи токсичность текста.
Ответь строго одним словом: TOXIC или NON_TOXIC.

Текст:
{text}
"""

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    r = requests.post(URL, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    with open("medvedev_news.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data:
        text = item["text"]
        label = label_text(text)

        print(f"{label} — {text[:100]}...")  # Печатаем первые 100 символов

        results.append({
            **item,
            "toxicity": label
        })

        time.sleep(0.3)  # не спамим API

    with open("labeled_news.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Обработано {len(results)} новостей, сохранено в labeled_news.json")


if __name__ == "__main__":
    main()
