import evaluate
from typing import List


class Metric():
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

    def compute_metrics(self, ground_truth: List[str], prediction: List[str]):
        res_b = self.bleu.compute(predictions=prediction, references=ground_truth)
        res_r = self.rouge.compute(predictions=prediction, references=ground_truth)
        res_m = self.meteor.compute(predictions=prediction, references=ground_truth)
        return {'bleu': res_b, 'rouge': res_r, 'meteor': res_m}