from typing import Dict, Tuple, List
import re
import warnings
import joblib
import pandas as pd
import pymorphy3
from collections import Counter

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class WaterAnalyzer:
    def __init__(self, model_path: str = "ruber_quality_model.pkl"):
        self.model = joblib.load(model_path)
        self.morph = pymorphy3.MorphAnalyzer()

        self.feature_names = [
            "readability_index",
            "adj_ratio",
            "adv_ratio",
            "repetition_ratio",
        ]
    
    def count_syllables(self, word: str) -> int:
        vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
        return sum(1 for char in word if char in vowels)
    
    def analyze_text_simple(self, text: str) -> Tuple[int, int, int]:
        raw_sentences = re.split(r'[.!?…]+', text)
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
        syllables = 0
        for word in words:
            normal_word = self.morph.parse(word)[0].normal_form
            syllables += self.count_syllables(normal_word)
        cleaned_sentences = []
        for s in raw_sentences:
            if s.strip():
                cleaned_sentences.append(s.strip())
        return len(cleaned_sentences), len(words), syllables
    
    def readability_index(self, text: str) -> float:
        sentences, words, syllables = self.analyze_text_simple(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        index = 206.835 - 1.3 * (words / sentences) - 60.1 * (syllables / words)
        return round(index, 2)
    
    def pos_ratios(self, text: str) -> Tuple[float, float]:
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
        pos = Counter()
        
        for w in words:
            p = self.morph.parse(w)[0].tag.POS
            pos[p] += 1
        
        total = sum(pos.values())
        if total == 0:
            return 0.0, 0.0
        
        adj = pos.get("ADJF", 0) + pos.get("ADJS", 0)
        adv = pos.get("ADVB", 0)
        
        return adj / total, adv / total
    
    def repetition_ratio(self, text: str) -> float:
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())
        if not words:
            return 0.0
        
        counts = Counter(words)
        return max(counts.values()) / len(words)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        readability = self.readability_index(text)
        adj_r, adv_r = self.pos_ratios(text)
        rep_r = self.repetition_ratio(text)
        
        return {
            "readability_index": readability,
            "adj_ratio": adj_r,
            "adv_ratio": adv_r,
            "repetition_ratio": rep_r
        }
    
    def predict(self, text: str, return_proba: bool = False) -> Dict:
        features = self.extract_features(text)
        X = pd.DataFrame([features])[self.feature_names].values
        
        try:
            proba_all = self.model.predict(X)
            water_proba = float(proba_all[0])
        except Exception:
            proba_all = self.model.predict_proba(X)[0]
            water_proba = float(proba_all[0]) if len(proba_all) > 0 else 0.0

        water_proba = max(0.0, min(1.0, water_proba))
        is_water = bool(water_proba >= 0.5)
        label = "ВОДА" if is_water else "НЕ ВОДА"

        result = {
            "text": text,
            "is_water": is_water,
            "water_label": label,
            "confidence": water_proba,
            "water_percentage": water_proba * 100,
            "features": features
        }
        
        if return_proba:
            result["probabilities"] = {
                "water": water_proba,
                "not_water": 1 - water_proba,
            }
        
        return result
    
    def interpret_features(self, features: Dict[str, float]) -> Dict[str, str]:
        interpretations = {}
        
        ri = features["readability_index"]
        if ri > 80:
            interpretations["readability"] = "очень легко читается"
        elif ri > 60:
            interpretations["readability"] = "нормально читается"
        elif ri > 40:
            interpretations["readability"] = "тяжеловато читается"
        else:
            interpretations["readability"] = "сложно читается"
        
        adj = features["adj_ratio"]
        if adj < 0.12:
            interpretations["adjectives"] = "фактология"
        elif adj < 0.18:
            interpretations["adjectives"] = "нейтрально"
        else:
            interpretations["adjectives"] = "возможная вода"
        
        adv = features["adv_ratio"]
        if adv < 0.03:
            interpretations["adverbs"] = "сухой текст"
        elif adv < 0.07:
            interpretations["adverbs"] = "нормально"
        else:
            interpretations["adverbs"] = "эмоциональная вода"
        
        rep = features["repetition_ratio"]
        if rep < 0.05:
            interpretations["repetitions"] = "хорошо"
        elif rep < 0.1:
            interpretations["repetitions"] = "терпимо"
        else:
            interpretations["repetitions"] = "вода"
        
        return interpretations
    
    def analyze(self, text: str, detailed: bool = True) -> Dict:
        result = self.predict(text, return_proba=True)
        
        if detailed:
            result["interpretations"] = self.interpret_features(result["features"])
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            results.append(self.analyze(text, detailed=True))
        return results
    
    def analyze_csv(self, csv_path: str, text_column: str = "text", 
                    output_path: str = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")
        
        results = []
        for text in df[text_column]:
            result = self.predict(text, return_proba=True)
            results.append(result)
        
        is_water_values = []
        water_label_values = []
        confidence_values = []
        water_prob_values = []
        for r in results:
            is_water_values.append(r["is_water"])
            water_label_values.append(r["water_label"])
            confidence_values.append(r["confidence"])
            water_prob_values.append(r["probabilities"]["water"])
        df["is_water"] = is_water_values
        df["water_label"] = water_label_values
        df["confidence"] = confidence_values
        df["water_probability"] = water_prob_values
        for feature in self.feature_names:
            feature_values = []
            for r in results:
                feature_values.append(r["features"][feature])
            df[feature] = feature_values
        
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return df
