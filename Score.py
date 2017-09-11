from nltk.tokenize import RegexpTokenizer
import numpy as np
import math


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
