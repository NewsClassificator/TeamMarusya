from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RuBERTSentimentAnalyzer:

    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny-sentiment-balanced",
        max_length: int = 512,
        batch_size: int = 32,
        temperature: float = 1.0,
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.label_mapping = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE",
        }

    def update_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)

    def get_parameters(self) -> Dict:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
        }

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_length - 2:
            return [text]

        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks

    def predict_sentiment(self, text: str) -> Dict[str, any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        moved_inputs = {}
        for k, v in inputs.items():
            moved_inputs[k] = v.to(self.device)
        inputs = moved_inputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            scaled_logits = outputs.logits / self.temperature
            predictions = torch.nn.functional.softmax(scaled_logits, dim=-1)

        probs = predictions.cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        predicted_label = (
            self.label_mapping[predicted_class]
            if confidence >= self.confidence_threshold
            else "UNCERTAIN"
        )

        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": {
                "NEGATIVE": float(probs[0]),
                "NEUTRAL": float(probs[1]),
                "POSITIVE": float(probs[2]),
            },
            "is_uncertain": confidence < self.confidence_threshold,
        }

    def predict_sentiment_with_chunking(self, text: str) -> Dict[str, any]:
        chunks = self.chunk_text(text)
        if len(chunks) == 1:
            result = self.predict_sentiment(text)
            result["chunks_used"] = 1
            result["chunking_details"] = None
            return result

        chunk_results = []
        for chunk in chunks:
            chunk_results.append(self.predict_sentiment(chunk))
        final_result = self.aggregate_chunk_results(chunk_results, text)
        final_result["chunks_used"] = len(chunks)
        chunk_details = []
        index_counter = 0
        for r in chunk_results:
            chunk_details.append(
                {
                    "chunk_index": index_counter,
                    "label": r["predicted_label"],
                    "confidence": r["confidence"],
                }
            )
            index_counter += 1
        final_result["chunking_details"] = {"chunk_results": chunk_details}
        return final_result

    def aggregate_chunk_results(
        self, chunk_results: List[Dict], original_text: str
    ) -> Dict[str, any]:
        from collections import Counter

        valid_predictions = []
        valid_confidences = []
        for result in chunk_results:
            if result["predicted_label"] != "UNCERTAIN":
                valid_predictions.append(result["predicted_label"])
                valid_confidences.append(result["confidence"])

        if not valid_predictions:
            confidence_values = []
            for r in chunk_results:
                confidence_values.append(r["confidence"])
            avg_confidence = np.mean(confidence_values)
            return {
                "text": original_text,
                "predicted_label": "UNCERTAIN",
                "confidence": float(avg_confidence),
                "probabilities": {
                    "NEGATIVE": 0.33,
                    "NEUTRAL": 0.34,
                    "POSITIVE": 0.33,
                },
                "is_uncertain": True,
            }

        vote_counts = Counter(valid_predictions)
        most_common_label = vote_counts.most_common(1)[0][0]
        same_label_confidences = []
        for idx in range(len(valid_predictions)):
            pred = valid_predictions[idx]
            conf = valid_confidences[idx]
            if pred == most_common_label:
                same_label_confidences.append(conf)
        avg_confidence = float(np.mean(same_label_confidences))

        agg_probs = {"NEGATIVE": 0.0, "NEUTRAL": 0.0, "POSITIVE": 0.0}
        for result in chunk_results:
            if result["predicted_label"] != "UNCERTAIN":
                for label in agg_probs:
                    agg_probs[label] += result["probabilities"][label]

        total_valid_chunks = len(valid_predictions)
        for label in agg_probs:
            agg_probs[label] /= total_valid_chunks

        final_label = (
            most_common_label
            if avg_confidence >= self.confidence_threshold
            else "UNCERTAIN"
        )

        return {
            "text": original_text,
            "predicted_label": final_label,
            "confidence": avg_confidence,
            "probabilities": agg_probs,
            "is_uncertain": avg_confidence < self.confidence_threshold,
            "voting_details": {
                "votes": dict(vote_counts),
                "total_chunks": len(chunk_results),
                "valid_chunks": total_valid_chunks,
                "winner": most_common_label,
                "winner_votes": vote_counts[most_common_label],
            },
        }
