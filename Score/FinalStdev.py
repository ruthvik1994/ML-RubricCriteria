import math


class FinalStdev(object):
    @staticmethod
    def calculate(means, sds, num_responses, total_mean):
        run = 0
        for i in range(len(num_responses)):
            run += (((num_responses[i] - 1) * sds[i] ** 2) + (num_responses[i] * (means[i] - total_mean) ** 2))
        final_stdev = round(math.sqrt((run * 1.0) / (sum(num_responses) - 1)), 3)
        return final_stdev
