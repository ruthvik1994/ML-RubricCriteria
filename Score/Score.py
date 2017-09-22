from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import numpy as np


class Score(object):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    helperwords = {word.strip("\n") for word in open("data/helperwords.txt", "r")}






