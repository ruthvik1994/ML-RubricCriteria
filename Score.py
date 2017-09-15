from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import numpy as np
import math


class ScoreEval1(object):

    @staticmethod
    def score(artifacts, metric_function):
        pass


class ScoreEval4(object):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    helperwords = {word.strip("\n") for word in open("data/helperwords.txt", "r")}

    @staticmethod
    def score(artifacts, metric, statistic):
        artifact_temp, means = list(), list()
        stdevs, num_comments = list(), list()
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
            num_comments.append(np.size(artifact_temp))
            del artifact_temp[:]
        if statistic == "mean":
            return FinalMean.calculate(means, num_comments)
        elif statistic == "stdev":
            return FinalStdev.calculate(means, stdevs, num_comments, FinalMean.calculate(means, num_comments))

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
