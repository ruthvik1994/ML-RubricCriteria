from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import nltk.data

import pickle
import os.path


def tag_wn(tag):
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    return None


def tag_to_synset(word, tag):
    tag = tag_wn(tag)
    if tag:
        try:
            return wn.synsets(word, tag)[0]
        except:
            return None


def get_similarity(sentence1, sentence2):
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tag_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tag_to_synset(*tagged_word) for tagged_word in sentence2]

    synsets1 = [word for word in synsets1 if word]
    synsets2 = [word for word in synsets2 if word]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = None
        temp = [synset.path_similarity(ss) for ss in synsets2]
        if temp:
            best_score = max(temp)
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    if count != 0:
        score /= count
    return score


def main():

    critviz_data = open(os.path.dirname(__file__) + '../data/rubrics_cv_eval4.pkl')
    expertiza_data = open(os.path.dirname(__file__) + '../data/rubrics_ez_eval4.pkl')
    rubrics1 = pickle.load(critviz_data)
    rubrics2 = pickle.load(expertiza_data)
    rubrics1.update(rubrics2)
    j = 0
    for rubric_id in rubrics1:
        desc = rubrics1[rubric_id]['desc']
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(desc)
        if len(sentences) < 2:
            print('-------------------------------------')
            print(sentences, 'Root')
        else:
            i = 1
            root = sentences[0]
            other = sentences[1:]
            print('Root : ' + root)
            for sentence in other:
                print(str(i)+' '+sentence+' : '+str(get_similarity(root, sentence)))
                i += 1


main()

