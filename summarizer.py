from nltk.corpus import stopwords
import nltk
import numpy as np
import sys
import re

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
            if w not in bow and w not in stop_words:
                bow.append(w) 
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
    for row, word in enumerate(bow):
        for col, sent in enumerate(sentences):
            word_sent_matrix[row][col] = 0
            # use count() can miscount words such as to
            # took, too and to will be counted toward the result
            # if the key is to => num of occurences of to = 3
            # even sentence contains only 1 to
            word_sent_matrix[row][col] = countNumOccurences(word, sent)



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
                word_sent_matrix[row][col] = countNumOccurences(word, sent) # TODO: assign 0

if not stopwords:
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

stop_words = stopwords.words('english')



fn = sys.argv[1]
top_k = int(sys.argv[2])
cell_value = sys.argv[3]

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
    
#print(word_sent_matrix[0])
#print(bow)


U,Sigma,V = np.linalg.svd(word_sent_matrix)
Vt = V.T

print(Vt)

# for i in range(top_k):
#     print(np.argmax(Vt[i]))
#     print(raw_sentences[np.argmax(Vt[i])])

