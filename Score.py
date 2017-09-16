from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import numpy as np
import math


class ScoreEval1(object):

    @staticmethod
    def score(artifacts, metric=None, statistic="stdev"):
        means = list()
        stdevs, num_responses = list(), list()
        for artifact_id in artifacts:
            means.append(np.mean(artifacts[artifact_id]))
            stdevs.append(np.std(artifacts[artifact_id]))
            num_responses.append(len(artifacts[artifact_id]))
        if statistic == "mean":
            return FinalMean.calculate(means, num_responses)
        elif statistic == "stdev":
            return FinalStdev.calculate(means, stdevs, num_responses, FinalMean.calculate(means, num_responses))


class ScoreEval4(object):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    helperwords = {word.strip("\n") for word in open("data/helperwords.txt", "r")}

    @staticmethod
    def score(artifacts, metric="reviewlen", statistic="mean"):
        artifact_temp, means = list(), list()
        stdevs, num_responses = list(), list()
        if metric == "reviewlen":
            metric_fun = ScoreEval4.revlength
        elif metric == "suggestive":
            metric_fun = ScoreEval4.suggestive
        for id in artifacts:
            for comment in artifacts[id]:
                tokens = ScoreEval4.tokenizer.tokenize(comment)
                if len(tokens) <= 5:
                    continue
                artifact_temp.append(metric_fun(tokens))
            means.append(np.mean(artifact_temp))
            stdevs.append(np.std(artifact_temp))
            num_responses.append(np.size(artifact_temp))
            del artifact_temp[:]
        if statistic == "mean":
            return FinalMean.calculate(means, num_responses)
        elif statistic == "stdev":
            return FinalStdev.calculate(means, stdevs, num_responses, FinalMean.calculate(means, num_responses))

    @staticmethod
    def revlength(tokens):
        return len(tokens)

    @staticmethod
    def suggestive(tokens):
        count = 0
        for token in tokens:
            if token in ScoreEval4.helperwords:
                count += 1
        return count


class FinalMean(object):
    @staticmethod
    def calculate(means, num_responses):
        run = 0
        for i in range(len(num_responses)):
            run += (means[i] * num_responses[i])
        total = round(run * 1.0 / sum(num_responses), 3)
        return total


class FinalStdev(object):
    @staticmethod
    def calculate(means, sds, num_responses, total_mean):
        run = 0
        for i in range(len(num_responses)):
            run += (((num_responses[i] - 1) * sds[i] ** 2) + (num_responses[i] * (means[i] - total_mean) ** 2))
        final_stdev = round(math.sqrt((run * 1.0) / (sum(num_responses) - 1)), 3)
        return final_stdev
