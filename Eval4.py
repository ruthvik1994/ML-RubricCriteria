import numpy as np
from Score import *


class Eval4(object):
    def __init__(self, data):
        self.data = data
        self.metrics = None
        self.labeled_data = None

    def find_labels(self, method=None):
        self.labeled_data = dict()
        revlen_means = list()
        for key, value in self.metrics.iteritems():
            revlen_means.append(value['mean_revlength'])
        if method == "binary":
            class_judge = np.percentile(revlen_means, 50)
            for key, value in self.metrics.iteritems():
                if value['mean_revlength'] > class_judge:
                    self.labeled_data[key] = 1
                else:
                    self.labeled_data[key] = 0
        else:
            mini_len = min(revlen_means)
            max_len = max(revlen_means)
            for key, value in self.metrics.iteritems():
                self.labeled_data[key] = (value['mean_revlength'] - mini_len) / (max_len - mini_len)

    def calculate_metrics(self):
        self.metrics = dict()
        for rubric_id in self.data:
            mean_revlength, sd_revlength = Score.score(self.data[rubric_id]['artifacts'], Score.revlength)
            mean_helpwords, sd_helpwords = Score.score(self.data[rubric_id]['artifacts'], Score.helpwords)
            if math.isnan(mean_revlength) or math.isnan(mean_helpwords):
                continue
            self.metrics[rubric_id] = {}
            self.metrics[rubric_id]['mean_revlength'], self.metrics[rubric_id][
                'sd_revlength'] = mean_revlength, sd_revlength
            self.metrics[rubric_id]['mean_sug'], self.metrics[rubric_id][
                'sd_sug'] = mean_helpwords, sd_helpwords




