from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from Score import Score


class Doc2Vec(object):
    def __init__(self, rubrics, min_count=1, window=5, sample=0, size=400, iter=15):
        self.model = Doc2Vec(min_count=min_count, window=window, sample=sample, size=size, iter=iter())
        self.X = []
        for key, value in rubrics.iteritems():
            words = Score.tokenizer.tokenize(rubrics[key]['desc'])
            artifacts = rubrics[key]['artifacts']
            for aritfact_id in artifacts:
                for comment in artifacts[aritfact_id]:
                    words += Score.tokenizer.tokenize(comment)
            for i in range(len(words)):
                words[i] = Score.stemmer.stem(words[i])
            self.X.append(words)

    

