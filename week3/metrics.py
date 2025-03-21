import evaluate
from typing import List


class Metric():
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

    def compute_metrics(self, ground_truth: List[List[str]], prediction: List[str]):
        ground_truth = [['A child in a pink dress is climbing up a set of stairs in an entry way .', 'A girl going into a wooden building .']]
        prediction = ['A girl goes into a wooden building .']

        res_b = self.bleu.compute(predictions=prediction, references=ground_truth)
        res_r = self.rouge.compute(predictions=prediction, references=ground_truth)
        res_m = self.meteor.compute(predictions=prediction, references=ground_truth)

        return res_b, res_r, res_m