
class FinalMean(object):
    @staticmethod
    def calculate(means, num_responses):
        run = 0
        for i in range(len(num_responses)):
            run += (means[i] * num_responses[i])
        total = round(run * 1.0 / sum(num_responses), 3)
        return total
