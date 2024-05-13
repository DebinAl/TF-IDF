import string
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

def pre_procesing(document, language_option = None):
    porter_stemmer = PorterStemmer()
    nltk_stopwords = set(stopwords.words(language_option))
    stemmer = StemmerFactory().create_stemmer() 
    document = str(document).casefold()                       #case folding    
    document = [sentence for sentence in document.split('.')] #split text

    for index, sentence in enumerate(document):
        arr_sentence = []
        sentence = str(sentence).translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) #punctuation
        for word in sentence.split():                                                                        #tokenisasi
            if language_option == 'indonesian':
                word = stemmer.stem(word)                                                                        #stemming
            elif language_option == 'english':
                word = porter_stemmer.stem(word)
            if word not in nltk_stopwords:                                                                   #stopword
                arr_sentence.append(word)
        document[index] = arr_sentence
    
    return document[:-1]

def term_list(pre_procesed):
    word_li = []
    for term in pre_procesed:
        for word in term:
            if word not in word_li:
                word_li.append(word)
    return word_li
        
def tf(document, word_list):
    tf_in_doc = []
    for word in word_list:
        word_freq = []
        for sentence in document:
            if word in sentence:
                word_freq.append(1)
            else:
                word_freq.append(0)
        tf_in_doc.append(word_freq)    
        
    return tf_in_doc

def boolean_retrieval(include, exclude, tf, word_list=[]):
    
    in_tf_index = []
    for in_query in include:
        in_tf_index.append(tf[word_list.index(in_query)])
        
    print("include val = " , in_tf_index)
    
    ex_tf_index = []
    for ex_query in exclude:
        ex_tf_index.append(tf[word_list.index(ex_query)])
    
    print("include val = ", ex_tf_index)
    
    temp = None
    for tfin in in_tf_index:
        if temp is None:
            temp = tfin
        else:
            temp = [x and y for x, y in zip(tfin, temp)]      
            
    for tfex in ex_tf_index:
        flip_val = []
        for val in tfex:
            flip_val.append(0 if val > 0 else 1)
        temp = [x and y for x, y in zip(temp, flip_val)]
        
    document_index = []
    for index, temp_val in enumerate(temp):
        if (temp_val!= 0):
            document_index.append(index)
    
    return document_index
    
def getText(document, index):
    document = [sentence for sentence in document.split('.')] #split text
    for val in index:
        print("Kalimat ", val+1)
        print(document[val])

    

#dataset (example)
id_language = "indonesian"
id_language_test = 'Sistem adalah kumpulan Sistem elemen. Adalah kumpulan elemen yang saling kumpulan berinteraksi . Sistem berinteraksi untuk mencapai tujuan.'

#pre processing
pre_procesing_result = pre_procesing(id_language_test, id_language)#udah bener
word_list_result = term_list(pre_procesing_result)#udah bener

#TF-IDF
tf_result = tf(pre_procesing_result, word_list_result)#udah bener

#boolean
include_query = ['sistem', 'kumpul']
exclude_query = ['interaksi']
                                                                                                                                                                                    
boolean_retrieval_index = boolean_retrieval(include_query, exclude_query, tf_result, word_list_result)
getText(id_language_test, boolean_retrieval_index)

