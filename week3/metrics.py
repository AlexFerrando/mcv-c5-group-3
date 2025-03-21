import evaluate
from typing import List
import nltk

class Metric():
    def __init__(self):
        self._download_nltk_resources()
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        
    def _download_nltk_resources(self):
        """Download NLTK resources quietly"""
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)

    def compute_metrics(self, ground_truth: List[str], prediction: List[str]):
        res_b = self.bleu.compute(predictions=prediction, references=ground_truth)
        res_r = self.rouge.compute(predictions=prediction, references=ground_truth)
        res_m = self.meteor.compute(predictions=prediction, references=ground_truth)

        return {'bleu': res_b, 'rouge': res_r, 'meteor': res_m}
