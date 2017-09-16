import pickle
from Eval4 import Eval4
from Eval1 import Eval1
from Doc2Vec import MyDoc2Vec


def main():

    """
    # Qualitative rubrics testing
    critviz_data = open("data/rubrics_cv_eval4.pkl")
    expertiza_data = open("data/rubrics_ez_eval4.pkl")
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    evaluator = Eval4(data=rubrics1)
    evaluator.calculate_metrics(metric="reviewlen", statistic="mean")
    evaluator.find_labels()
    for key in evaluator.labeled_data:
        print[key, evaluator.data[key]['desc'], evaluator.metrics[key], evaluator.labeled_data[key]]
        print

    """

    # Quantitative rubrics testing
    critviz_data = open("data/rubrics_cv_eval1.pkl")
    expertiza_data = open("data/rubrics_ez_eval1.pkl")
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    evaluator = Eval1(data=rubrics1)
    evaluator.calculate_metrics()
    evaluator.find_labels()
    for key in evaluator.labeled_data:
        print(key, evaluator.data[key]['desc'], evaluator.metrics[key], evaluator.labeled_data[key])
        print

    """
    # doc2vec = MyDoc2Vec(rubrics=evaluator.data)
    # doc2vec.build_vectors()
    
    """


if __name__ == '__main__':
    main()
