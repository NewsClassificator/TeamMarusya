"""Инференс для определения кликбейтных заголовков без CLI и отладочных выводов."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class ClickbaitDetector:
    """Детектор кликбейтных заголовков."""

    def __init__(self, model_path: str = "my_awesome_model"):
        import os

        if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
            checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                model_path = os.path.join(model_path, latest_checkpoint)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # CPU
        )

    def predict(self, text: str):
        """Предсказание для одного заголовка."""
        result = self.classifier(text)[0]
        return result

    def predict_batch(self, texts):
        """Предсказание для списка заголовков."""
        return self.classifier(texts)

    def is_clickbait(self, text: str, threshold: float = 0.5) -> bool:
        """Возвращает True, если заголовок кликбейтный по заданному порогу."""
        result = self.predict(text)
        return result["label"] == "кликбейт" and result["score"] >= threshold
