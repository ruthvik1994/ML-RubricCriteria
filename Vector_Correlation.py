from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random
from Score import Score
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import uuid
from sklearn.metrics.pairwise import cosine_similarity


class VectorSimilarity(object):

    def __init__(self, rubrics, min_count=1, window=5, sample=0, size=50, iter=15):
        self.model = Doc2Vec(min_count=min_count, window=window, sample=sample, size=size, iter=iter)
        self.vectors = dict()
        self.sentences = list()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = PorterStemmer()
        self.rubric_commentid = {}
        count = 1
        for key, value in rubrics.iteritems():
            self.rubric_commentid[key] = []
            self.sentences.append(LabeledSentence(self.stem(rubrics[key]['desc']), [key]))
            artifacts = rubrics[key]['artifacts']
            for artifact_id in artifacts:
                for comment in artifacts[artifact_id]:
                    id = "C"+str(count)
                    self.rubric_commentid[key].append(id)
                    self.sentences.append(LabeledSentence(self.stem(comment), [id]))
                    count += 1

    def stem(self, sentence):
        words = self.tokenizer.tokenize(sentence)
        for i in range(len(words)):
            words[i] = self.stemmer.stem(words[i])
        return words

    def build_vectors(self):
        self.model.build_vocab(self.sentences)
        print("Building Vectors .... ")
        for i in range(3):
            print("Iteration : "+str(i))
            random.shuffle(self.sentences)
            self.model.train(self.sentences, total_examples=len(self.sentences), epochs=10)
        for label in self.model.docvecs.doctags:
            self.vectors[label] = self.model.docvecs[label]

    def find_similarities(self):
        similarities = {}
        for rubric_id in self.rubric_commentid:
            rubric_vector = self.vectors[rubric_id]
            similarities[rubric_id] = []
            for comment_id in self.rubric_commentid[rubric_id]:
                comment_vector = self.vectors[comment_id]
                cos_similarity = cosine_similarity(rubric_vector, comment_vector)
                similarities[rubric_id].append(round(cos_similarity[0][0], 2))
        return similarities
