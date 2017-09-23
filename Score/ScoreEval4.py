import numpy as np
from Score import Score
from .FinalMean import FinalMean
from .FinalStdev import FinalStdev


class ScoreEval4(Score):

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

