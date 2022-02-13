from nltk.corpus import stopwords
import nltk
import numpy as np
import sys
import re

def readFile(fn):
    with open(fn, "r") as file:
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
        while '' in temp:
            temp.remove('')
        original_sentences += temp
        for i in range(len(temp)):
            temp[i] = re.sub(r'[^\w\s]','',temp[i]).lower()
        clean_sentences += temp
    return original_sentences, clean_sentences

def createBagOfWords(sentences):    
    bow = []
    for s in sentences:
        words = s.split(" ")
        for w in words:
            if w not in stop_words and w not in bow:
                bow.append(w) 
    return bow

#a matrix representation where columns are sentences and rows are words
def buildWordSentenceMatrix(bow, sentences):
    word_sent_matrix = np.zeros((len(bow), len(sentences)))
    for row, word in enumerate(bow):
        for col, sent in enumerate(sentences):
            word_sent_matrix[row][col] = sent.count(word)
    return word_sent_matrix

if not stopwords:
    nltk.download('stopwords')
stop_words = stopwords.words('english')
fn = sys.argv[1]
top_k = int(sys.argv[2])

raw_sentences, clean_sentences = getSentences(readFile(fn))
bow = createBagOfWords(clean_sentences)
word_sent_matrix = buildWordSentenceMatrix(bow, clean_sentences)
U,Sigma,V = np.linalg.svd(word_sent_matrix, full_matrices=False)

Vt = V.T
for i in range(top_k):
    print(np.argmax(Vt[i]))
    print(raw_sentences[np.argmax(Vt[i])])

