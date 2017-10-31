import pickle
from Eval4 import Eval4
from Eval1 import Eval1
from Doc2Vec import MyDoc2Vec
from Vector_Correlation import VectorSimilarity as VS
import csv
import numpy as np

def main():


    # Qualitative rubrics testing
    critviz_data = open("data/rubrics_cv_eval4.pkl")
    expertiza_data = open("data/rubrics_ez_eval4.pkl")
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    """
    evaluator = Eval4(data=rubrics1)

    evaluator.calculate_metrics(metric="reviewlen", statistic="mean")
    evaluator.find_labels()
    for key in evaluator.labeled_data:
        print[key, evaluator.data[key]['desc'], evaluator.metrics[key], evaluator.labeled_data[key]]
        print


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


    doc2vec = MyDoc2Vec(rubrics=evaluator.data)
    doc2vec.build_vectors()
    modelname = "d2v_qual"
    doc2vec.model.save(modelname)
    
    """
    simfile = open("similarities.csv", "w")
    fieldnames = ['CriterionID', 'Description', 'MeanSimilarity']
    writer = csv.DictWriter(simfile, fieldnames=fieldnames)
    writer.writeheader()
    vec_similarity = VS(rubrics=rubrics1)
    vec_similarity.build_vectors()
    similarities = vec_similarity.find_similarities()
    for each in similarities:
        writer.writerow({'CriterionID': each, 'Description': rubrics1[each]['desc'],
                         'MeanSimilarity': np.mean(similarities[each])})

if __name__ == '__main__':
    main()
