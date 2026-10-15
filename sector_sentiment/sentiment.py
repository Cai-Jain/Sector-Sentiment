"""Sentiment classifiers.

FinBERT (ProsusAI/finbert) is the default: a BERT model fine-tuned on financial
news, so it knows "shares fall on guidance cut" is negative and "beats
estimates" is positive, which generic models often miss. VADER is a fast,
dependency-light fallback.

Every classifier returns, per text: label, P(positive), P(negative), P(neutral),
and a polarity score = P(pos) - P(neg) in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sentiment:
    label: str
    positive: float
    negative: float
    neutral: float

    @property
    def score(self) -> float:
        return self.positive - self.negative


class FinBERT:
    MODEL = "ProsusAI/finbert"

    def __init__(self, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL).eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]

    def classify(self, texts: list[str]) -> list[Sentiment]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for p in probs:
                d = dict(zip(self.labels, p))
                out.append(Sentiment(max(d, key=d.get), d["positive"], d["negative"], d["neutral"]))
        return out


class Vader:
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def classify(self, texts: list[str]) -> list[Sentiment]:
        out = []
        for t in texts:
            s = self.analyzer.polarity_scores(t)
            label = "positive" if s["compound"] >= 0.05 else "negative" if s["compound"] <= -0.05 else "neutral"
            out.append(Sentiment(label, s["pos"], s["neg"], s["neu"]))
        return out


def get_classifier(name: str):
    return {"finbert": FinBERT, "vader": Vader}[name]()
