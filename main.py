import pickle
from Eval4 import Eval4


def main():
    critviz_data = open("data/rubrics_cr_eval4.pkl")
    expertiza_data = open("data/rubrics_ez_eval4.pkl")
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    evaluator = Eval4(data=rubrics1)
    print("Total number of rubric criteria before cleaning : %d" % len(evaluator.data))
    evaluator.calculate_metrics()
    print("Total number of rubric criteria after cleaning : %d" % len(evaluator.metrics))
    evaluator.find_labels()
    '''
    for key in evaluator.labeled_data:
        print[key, evaluator.data[key]['desc'], evaluator.metrics[key]['mean_revlength'], evaluator.labeled_data[key]]
        print
    '''

if __name__ == '__main__':
    main()