import pickle
from pathlib import Path


def main() -> None:
    model_path = Path(__file__).with_name("logreg_water_model.pkl")
    if not model_path.is_file():
        print(f"Model file not found: {model_path}")
        return

    with model_path.open("rb") as f:
        model = pickle.load(f)

    print("Loaded object type:", type(model))

    texts = [
        "В районе ожидаются сильные дожди, возможны наводнения.",
        "Сегодня солнечно и без осадков, уровень воды нормальный.",
    ]
    preds = model.predict(texts)
    proba = getattr(model, "predict_proba", None)
    probs = proba(texts) if proba is not None else None

    for text, label, p in zip(texts, preds, probs if probs is not None else [None] * len(texts)):
        print("\nТекст:", text)
        print("Предсказанный класс:", label)
        if p is not None:
            print("Вероятности классов:", p)


if __name__ == "__main__":
    main()

