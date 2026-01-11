from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class ClickbaitDetector:

    def __init__(self, model_path: str = "my_awesome_model"):
        import os

        if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
            checkpoints = []
            for d in os.listdir(model_path):
                if d.startswith("checkpoint-"):
                    checkpoints.append(d)
            if checkpoints:
                latest_checkpoint = checkpoints[0]
                for item in checkpoints:
                    parts = item.split("-")
                    latest_parts = latest_checkpoint.split("-")
                    if len(parts) > 1 and len(latest_parts) > 1:
                        if parts[1].isdigit() and latest_parts[1].isdigit():
                            if int(parts[1]) > int(latest_parts[1]):
                                latest_checkpoint = item
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
        result = self.classifier(text)[0]
        return result

    def predict_batch(self, texts):
        return self.classifier(texts)

    def is_clickbait(self, text: str, threshold: float = 0.5) -> bool:
        result = self.predict(text)
        return result["label"] == "кликбейт" and result["score"] >= threshold
