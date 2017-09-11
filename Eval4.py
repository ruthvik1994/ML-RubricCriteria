import pickle
import math
import numpy as np
from nltk.tokenize import RegexpTokenizer


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


class Score(object):
    tokenizer = RegexpTokenizer(r'\w+')
    helperwords = {word.strip("\n") for word in open("data/helperwords.txt", "r")}

    @staticmethod
    def score(artifacts, metric_function):
        artifact_temp, means = list(), list()
        stdevs, num_comments = list(), list()
        for id in artifacts:
            for comment in artifacts[id]:
                tokens = Score.tokenizer.tokenize(comment)
                if len(tokens) <= 5:
                    continue
                artifact_temp.append(metric_function(tokens))
            means.append(np.mean(artifact_temp))
            stdevs.append(np.std(artifact_temp))
            num_comments.append(np.size(artifact_temp))
            del artifact_temp[:]
        total_mean = FinalMean.calculate(means, num_comments)
        total_sd = FinalStdev.calculate(means, stdevs, num_comments, total_mean)
        return total_mean, total_sd

    @staticmethod
    def revlength(tokens):
        return len(tokens)

    @staticmethod
    def helpwords(tokens):
        count = 0
        for token in tokens:
            if token in Score.helperwords:
                count += 1
        return count


class FinalMean(object):
    @staticmethod
    def calculate(means, num_comments):
        run = 0
        for i in range(len(num_comments)):
            run += (means[i] * num_comments[i])
        total = round(run * 1.0 / sum(num_comments), 3)
        return total


class FinalStdev(object):
    @staticmethod
    def calculate(means, sds, num_comments, total_mean):
        run = 0
        for i in range(len(num_comments)):
            run += (((num_comments[i] - 1) * sds[i] ** 2) + (num_comments[i] * (means[i] - total_mean) ** 2))
        final_stdev = round(math.sqrt((run * 1.0) / (sum(num_comments) - 1)), 3)
        return final_stdev


def main():
    critviz_data = open("data/rubrics_cr_eval4.pkl")
    expertiza_data = open("data/rubrics_ez_eval4.pkl")
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    evaluator = Eval4(data=rubrics1)
    print("Total number of rubric criteria before cleaning : %d" % len(evaluator.data))
    evaluator.calculate_metrics()
    print("Total number of rubric criteria after cleaning : %d" % len(evaluator.metrics))
    evaluator.find_labels()
    for key in evaluator.labeled_data:
        print[key, evaluator.data[key]['desc'], evaluator.metrics[key]['mean_revlength'], evaluator.labeled_data[key]]
        print

if __name__ == '__main__':
    main()
