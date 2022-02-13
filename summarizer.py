from nltk.corpus import stopwords
from collections import Counter
import nltk
import numpy as np
import sys
import re

from prometheus_client import Summary

def readFile(fn):
    with open(fn, "r", encoding="utf-8") as file:
        text = file.read()
        paragraphs = text.split("\n")
        while '' in paragraphs:
            paragraphs.remove('')
    return paragraphs

def getSentences(paragraphs):
    clean_sentences = []
    original_sentences = []
    for p in paragraphs:
        temp = p.split(". ")
        # Not to include short sentences such as 1.
        # Also, short sentences contain little information
        temp = [sent for sent in temp if len(sent) >= 3]
        original_sentences += temp
        for i in range(len(temp)):
            #TODO: Not to include quotation mark
            temp[i] = re.sub(r'[^\w\s]','',temp[i]).lower()
            
        clean_sentences += temp
    return original_sentences, clean_sentences

def createBagOfWords(sentences):    
    bow = []
    for s in sentences:
        words = s.strip().split(" ")
        for word in words:
            if word not in stop_words and word not in bow:
                bow.append(word)
    return bow

def countNumOccurences(word, sentence):
    words = sentence.split(" ")
    ret = 0
    for w in words:
        if w == word:
            ret += 1
    return ret

#a matrix representation where columns are sentences and rows are words
def buildFrequencyMatrix(word_sent_matrix, bow, sentences):
    sentMap = {}
    for row, word in enumerate(bow):
        for col, sent in enumerate(sentences):
            if not sentMap.get(col):
                sentMap[col] = Counter(sent.split(" "))
            word_sent_matrix[row][col] = sentMap[col].get(word,0)

def buildBinaryMatrix(word_sent_matrix, bow, sentences):
    for row, word in enumerate(bow):
        for col, sent in enumerate(sentences):
            if word in sent:
                word_sent_matrix[row][col] = 1

def buildNounsMatrix(word_sent_matrix, bow, sentences):
    for row, word in enumerate(bow):
        for col, sent in enumerate(sentences):
            tagged = nltk.pos_tag([word])[0][1]
            if tagged == 'NN' or tagged == 'NNS' or tagged == 'NNPS' or tagged == 'NNP':
                word_sent_matrix[row][col] = countNumOccurences(word, sent) # TODO: use 1 or 0

def summarizeText(Vt, raw_sentences, method, top_k):
    def GongLiuMethod(Vt, raw_sentences):
        chosen_sentences = []
        for i in range(top_k):
            chosen = np.argmax(Vt[i])
            chosen_sentences.append((chosen, raw_sentences[chosen]))
        return chosen_sentences
    
    def crossMethod(Vt, raw_sentences):
        chosen_sentences = []
        avg = np.mean(Vt, axis=1).reshape(-1, 1)
        zero_mask = Vt > avg
        Vt = np.where(zero_mask == True, Vt, 0)
        length = np.sum(Vt, axis=0)
        for i in range(top_k):
            chosen = np.argmax(length)
            length[chosen] = -sys.maxsize
            chosen_sentences.append((chosen, raw_sentences[chosen]))
        return chosen_sentences

    if method == "gongliu":
        chosen_sentences = GongLiuMethod(Vt, raw_sentences)
    elif method == "cross":
        chosen_sentences = crossMethod(Vt, raw_sentences)
    else:
        print("ERROR...CAN'T FIND METHOD")
        exit(-1)
    
    chosen_sentences = sorted(chosen_sentences, key= lambda pair:pair[0])
    summary = ""
    for pair in chosen_sentences:
        if pair[1][-1] != ".":
            summary += pair[1] + ". "
        else:
            summary += pair[1] + " "
    return summary

if not stopwords:
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

stop_words = stopwords.words('english')

fn = sys.argv[1]
top_k = int(sys.argv[2])
cell_value = sys.argv[3]
method = sys.argv[4]

raw_sentences, clean_sentences = getSentences(readFile(fn))
bow = createBagOfWords(clean_sentences)

word_sent_matrix = np.zeros((len(bow), len(raw_sentences)))

if cell_value == "freq":
    buildFrequencyMatrix(word_sent_matrix, bow, clean_sentences)
elif cell_value == "binary":
    buildBinaryMatrix(word_sent_matrix, bow, clean_sentences)
elif cell_value == "root":
    buildNounsMatrix(word_sent_matrix, bow, clean_sentences)
else:
    print("ERROR... CAN'T BUILD MATRIX")
    exit(-1)
    
U,Sigma,V = np.linalg.svd(word_sent_matrix)
Vt = V.T
#a little tweak for more accuracy
Vt[0, 0] *= -1

#TODO: why not run all algorithms and find the common sentences among them?

print(summarizeText(Vt, raw_sentences, method, top_k))

