import evaluate
from typing import List
import nltk

class Evaluator():
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

    def evaluate(self, ground_truth: List[str], prediction: List[str]):
        try:
            res_b1 = self.bleu.compute(predictions=prediction, references=ground_truth, max_order=1)
        except Exception as e:
            print(f"Error computing BLEU-1: {e}")
            res_b1 = 0
        
        try:
            res_b2 = self.bleu.compute(predictions=prediction, references=ground_truth, max_order=2)
        except Exception as e:
            print(f"Error computing BLEU-2: {e}")
            res_b2 = 0
        
        try:
            res_r = self.rouge.compute(predictions=prediction, references=ground_truth)
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            res_r = 0
            
        try:
            res_m = self.meteor.compute(predictions=prediction, references=ground_truth)
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            res_m = 0

        return {'bleu1': res_b1, 'bleu2': res_b2, 'rouge': res_r, 'meteor': res_m}
