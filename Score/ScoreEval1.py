from .FinalMean import FinalMean
from .FinalStdev import FinalStdev
from .FleissKappa import FleissKappa
import numpy as np


class ScoreEval1(object):

    @staticmethod
    def score(artifacts, metric=None, statistic="stdev"):
        means = list()
        stdevs, num_responses = list(), list()
        if statistic == "fleiss":
            kappa = FleissKappa(artifacts)
            return FleissKappa.compute(kappa.mat)
            # return kappa.compute(artifacts)
        for artifact_id in artifacts:
            means.append(np.mean(artifacts[artifact_id]))
            stdevs.append(np.std(artifacts[artifact_id]))
            num_responses.append(len(artifacts[artifact_id]))
        if statistic == "mean":
            return FinalMean.calculate(means, num_responses)
        elif statistic == "stdev":
            return FinalStdev.calculate(means, stdevs, num_responses, FinalMean.calculate(means, num_responses))
