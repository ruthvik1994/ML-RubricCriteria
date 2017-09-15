import pickle
from Eval4 import Eval4
from Doc2Vec import MyDoc2Vec


def main():
    critviz_data = open("data/rubrics_cv_eval4.pkl")
    expertiza_data = open("data/rubrics_ez_eval4.pkl")
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    evaluator = Eval4(data=rubrics1)
    evaluator.calculate_metrics(metric="reviewlen", statistic="mean")
    evaluator.find_labels()
    # doc2vec = MyDoc2Vec(rubrics=evaluator.data)
    # doc2vec.build_vectors()

    

if __name__ == '__main__':
    main()
