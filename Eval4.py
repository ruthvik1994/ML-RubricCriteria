import numpy as np
import math
from Score.ScoreEval4 import ScoreEval4 as Score


class Eval4(object):
    def __init__(self, data):
        self.data = data
        self.metrics = None
        self.labeled_data = None

    def find_labels(self, method=None):
        self.labeled_data = dict()
        scores = list()
        for key, value in self.metrics.iteritems():
            if value != -1:
                scores.append(value)
        if method == "binary":
            class_judge = np.percentile(scores, 50)
            for key, value in self.metrics.iteritems():
                if value > class_judge:
                    self.labeled_data[key] = 1
                else:
                    self.labeled_data[key] = 0
        else:
            mini_len = min(scores)
            max_len = max(scores)
            for key, value in self.metrics.iteritems():
                self.labeled_data[key] = (value - mini_len) / (max_len - mini_len)

    def calculate_metrics(self, metric=None, statistic=None):
        self.metrics = dict()
        for rubric_id in self.data:
            metric_score = Score.score(self.data[rubric_id]['artifacts'], metric, statistic)
            print(metric_score)
            if math.isnan(metric_score):
                self.metrics[rubric_id] = -1
                continue
            self.metrics[rubric_id] = metric_score




