import pickle
import math
import numpy as np
from nltk.tokenize import RegexpTokenizer


class Eval4(object):
    def __init__(self, data, helperwords):
        self.data = data
        self.helperwords = helperwords
        self.labeled_data = None

    def label(self):
        self.labeled_data = LabelRubrics.label(self.data, self.helperwords)


class LabelRubrics(object):

    tokenizer = RegexpTokenizer(r'\w+')
    np.seterr(divide='ignore', invalid='ignore')

    @staticmethod
    def label(rubrics, helper_words):
        # labels notes down the labels defined based on scores
        # mean_revlen stores the average of review lengths garnered by all artifacts under each rubric
        # mean_help     "   "   "   "    suggestions, modal verbs etc "     "   "       "       "   "
        # sd_revlen     "   "   "  standard deviation of review length "   "   "   "   "   "   "
        # sd_help     "   "   "   " "   "   "   "   " of suggestions, modal verbs etc "     "       "
        labels = dict()
        mean_revlen, sd_revlen = list(), list()
        mean_help, sd_help = list(), list()
        for rubric_id in rubrics:
            mean_revlength, sd_revlength = LabelRubrics.score_revlength(rubrics[rubric_id]['artifacts'])
            mean_helpwords, sd_helpwords = LabelRubrics.score_helpwords(rubrics[rubric_id]['artifacts'], helper_words)
            if math.isnan(mean_revlength) or math.isnan(mean_helpwords):
                continue
            mean_help.append(mean_helpwords)
            mean_revlen.append(mean_revlength)
            sd_help.append(sd_helpwords)
            sd_revlen.append(sd_revlength)
            labels[rubric_id] = {}
            labels[rubric_id]['mean_revlength'], labels[rubric_id][
                'sd_revlength'] = mean_revlength, sd_revlength
            labels[rubric_id]['mean_sug'], labels[rubric_id][
                'sd_sug'] = mean_helpwords, sd_helpwords
        judge = np.percentile(mean_revlen, 50)
        for key, value in labels.iteritems():
            if value['mean_revlength'] < judge:
                labels[key]['class'] = 0
            else:
                labels[key]['class'] = 1
        return labels

    @staticmethod
    def score_revlength(artifacts):
        artifact_temp, means = list(), list()
        stdevs, num_comments = list(), list()
        for id in artifacts:
            for comment in artifacts[id]:
                tokens = LabelRubrics.tokenizer.tokenize(comment)
                if len(tokens) <= 5:
                    continue
                artifact_temp.append(len(tokens))
            means.append(np.mean(artifact_temp))
            stdevs.append(np.std(artifact_temp))
            num_comments.append(np.size(artifact_temp))
            del artifact_temp[:]
        total_mean = LabelRubrics.final_mean(means, num_comments)
        total_sd = LabelRubrics.final_sd(means, stdevs, num_comments, total_mean)
        return total_mean, total_sd

    @staticmethod
    def score_helpwords(artifacts, helperwords):
        artifact_temp, means = list(), list()
        stdevs, num_comments = list(), list()
        for id in artifacts:
            for comment in artifacts[id]:
                tokens = LabelRubrics.tokenizer.tokenize(comment)
                if len(tokens) <= 5:
                    continue
                count = 0
                for token in tokens:
                    if token in helperwords:
                        count += 1
                artifact_temp.append(count)
            means.append(np.mean(artifact_temp))
            stdevs.append(np.std(artifact_temp))
            num_comments.append(np.size(artifact_temp))
            del artifact_temp[:]
        total_mean = FinalMean.final_mean(means, num_comments)
        total_sd = FinalStdev.final_sd(means, stdevs, num_comments, total_mean)
        return total_mean, total_sd


class FinalMean(object):

    @staticmethod
    def final_mean(means, num_comments):
        run = 0
        for i in range(len(num_comments)):
            run += (means[i] * num_comments[i])
        total = round(run * 1.0 / sum(num_comments), 3)
        return total


class FinalStdev(object):

    @staticmethod
    def final_sd(means, sds, num_comments, total_mean):
        run = 0
        for i in range(len(num_comments)):
            run += (((num_comments[i] - 1) * sds[i] ** 2) + (num_comments[i] * (means[i] - total_mean) ** 2))
        final_stdev = round(math.sqrt((run * 1.0) / (sum(num_comments) - 1)), 3)
        return final_stdev


def main():
    critviz_data = open("data/rubrics_cr_eval4.pkl")
    expertiza_data = open("data/rubrics_ez_eval4.pkl")
    helper_words = {word.strip("\n") for word in open("data/helperwords.txt", "r")}
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    evaluator = Eval4(data=rubrics1, helperwords=helper_words)
    print("Total number of rubric criteria before cleaning : %d" % len(evaluator.data))
    evaluator.label()
    print("Total number of rubric criteria after cleaning : %d" % len(evaluator.labeled_data))


if __name__ == '__main__':
    main()








