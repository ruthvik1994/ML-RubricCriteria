# Project Title

Machine Learning to assess the quality of Rubric Criteria

## Description
At NC state university, as a part of the course CSC 517 Object-Oriented Design and Development, students are expected to provide feedback in the form of either ranking, rating or text to other projects. Designing a good Rubric Criteria would be the first step to improve the inter-rater reliability measures and also make students give suggestive feedback to help peers improve their performance in the next evaluation.
There are a lot of research papers that explain how a Rubric should be designed and introduced to students but there was no platform that provides a method to assess how a rubric might perform or how students might interpret the feedback form (list of rubric criteria).
This package lets you compute those metrics that can advise how well your rubric criteria is received by the students

### Prerequisites
The package is so developed to compute according to the data format we created that could aggregate all the required classes into a chain of calls to finally compute.
1) Data Format:
```
rubrics = {'rubric_id1':{}, 'rubric_id2':{}, .............. }
'rubric_id1': {'desc':'Textual Description of the Rubric', 'artifacts': {}}
'artifacts': {'artifact_id1':[rating1, rating2, ....], 'artifact_id2':[rating1, rating2, .....], ..........}
```
Rubrics is a dictionary with keys as RubricsIDs and values a dictionary that has desc and artifacts as keys. Artifacts represents projects in evaluation and therefore have an artifact_id associated with them and value as a list of ratings from students that evaluated that artifact/project.

For Qualitative Type
```
'artifacts': {'artifact_id1':[feedbacktext1, feedbacktext2, ....]}
```
2) Packages
numpy, gensim, nltk

### Metrics
In order to compute inter-rater reliability measures or any statistic to assess the quality, the textual feedback is needed to be converted into a numerical value. There are two metrics available
```
1) Review Length: Students tend to write more if the Rubric is understood or demands. To enable peer reviews to help improvement, design of rubric needs to attempt increasing the response length
2) Suggestive content: Some rubrics effectively makes a student write suggestive feedback which is far more restrictive metric than review length.
```

### Statistic
There needs to be a measure that computes the overall performance of the rubric overall projects and responses. We have considered some that are applicable to our data
```
- Mean of the review length 
- Standard Deviation of the review length
- Fleiss Kappa (an inter-rater reliability measure)
- Intraclass correlation coefficient (Not integrated yet)
```
## How to use the package

### For Quantitative Rubrics (works with rating on an ordinal scale)
Create Eval1 object
```
eval1_obj = Eval1(data=rubrics)
eval1_obj.calculate_metrics(statistic='stdev')
or
eval1_obj.calculate_metrics(statistic='mean')
or
eval1_obj.calculate_metrics(statistic='fleiss')
```
### For Qualitative Rubrics (works for textual feedback)
Create Eval4 object
```
eval4_obj = Eval4(data=rubrics)
eval4_obj.calculate_metrics(metric='reviewLen', statistic='stdev')
or
eval4_obj.calculate_metrics(metric='reviewLen', statistic='mean')
or
eval4_obj.calculate_metrics(metric='reviewLen', statistic='fleiss')
```

### For Evaluation
The statistics calculated can be used to label the rubrics either with a normalized score or with a binary label.
```
eval4_obj.find_labels(method='binary')
or
eval4_obj.find_labels(method='regression')
```
Finally eval1_obj or eval4_obj has all the information as follows
```
eval1_obj.metrics = {'RubricID1': Score Computed, 'RubricID2': Score computed}
eval1_obj.labeled_data = {'RubricID1': Label or Normalized Score, 'RubricID2': Label or Normalized Score}
```

### Machine Learning Techniques
There is another class that computes vector representations using Doc2Vec technique so that the package can emit training data where each instance is in the form vector1: label1, vector2:label2 and therefore can be used to apply any supervised technique of your interest
```
doc2vec = MyDoc2Vec(rubrics=eval1_obj.data)
doc2vec.build_vectors()
```
To Form the training data
```
X, Y = []
for id, label in eval1_obj.labeled_data.iteritems:
 X.append(doc2vec.vectors[id])
 Y.append(label)
```
## Acknowledgments

* Dr. Edward Gehringer, Associate Professor, Computer Science and ECE, North Carolina State University
* Dr. Ferry Pramudianto, Research Scholar, North Carolina State University
