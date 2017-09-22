from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random
from Score import Score


class MyDoc2Vec(object):
    def __init__(self, rubrics, min_count=1, window=5, sample=0, size=400, iter=15):
        self.model = Doc2Vec(min_count=min_count, window=window, sample=sample, size=size, iter=iter)
        self.vectors = dict()
        self.sentences = list()
        for key, value in rubrics.iteritems():
            words = Score.tokenizer.tokenize(rubrics[key]['desc'])
            '''
            artifacts = rubrics[key]['artifacts']
            for artifact_id in artifacts:
                for comment in artifacts[artifact_id]:
                    words += Score.tokenizer.tokenize(comment) '''
            for i in range(len(words)):
                words[i] = Score.stemmer.stem(words[i])
            self.sentences.append(LabeledSentence(words, [key]))

    def build_vectors(self):
        self.model.build_vocab(self.sentences)
        for i in range(5):
            random.shuffle(self.sentences)
            self.model.train(self.sentences, total_examples=len(self.sentences), epochs=10)
        for label in self.model.docvecs.doctags:
            self.vectors[label] = self.model.docvecs[label]

