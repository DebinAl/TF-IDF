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
        
def tf(document, query_list):
    tf_in_doc = []
    for query in query_list:
        query_doc_freq = []
        for sentence in document:
            count = sum(1 for word in sentence if word == query)
            query_doc_freq.append(count)
        tf_in_doc.append(query_doc_freq)
    return tf_in_doc

def df(term_frequency):
    total_document_term = []
    for term in term_frequency:
        count = sum(1 for val in term if val > 0)
        total_document_term.append(count)
    return total_document_term

def ddf(document, df):
    doc_term_freq = []
    for val in df:
        doc_term_freq.append(len(document)/val)
    return doc_term_freq

def idf(ddf):
    inverse_doc_freq = []
    for val in ddf:
        inverse_doc_freq.append(math.log10(val))
    return inverse_doc_freq


def query_weight(word_list, find_query, idf, tf):
    query_weight = []
    for word, idf_val in zip(word_list, idf):
        if word in find_query :
            val = 1*idf_val
            query_weight.append(val)
        else:
            val = 0*idf_val
            query_weight.append(val)
    
    for tf_val in tf:
        print("dawdaw")
            
    
    return query_weight

def doc_weight(tf, idf):
    weight_by_Term = []
    for term_freq, doc_freq in zip(tf, idf):
        term_row = []
        for term_val in term_freq:
            term_row.append(term_val*doc_freq)
        weight_by_Term.append(term_row)
    return weight_by_Term
    
def query_distance(query_weight):
    square_val = []
    for weight in query_weight:
        square_val.append(weight**2) 
    return math.sqrt(sum(square_val))
  
    
def document_distance(doc_weight):
    square_val = [[] for _ in range(len(doc_weight[0]))]
    for weight_li in doc_weight:
        for index, weight in enumerate(weight_li):
            square_val[index].append(weight**2)
            
    doc_distance = []
    for val in square_val:
        doc_distance.append(math.sqrt(sum(val)))

    return doc_distance
    
    
def dot_product(query_weight, doc_weight):
    query_square_val = []
    for weight in query_weight:
        query_square_val.append(weight**2) 
    
    doc_square_val = [[] for _ in range(len(doc_weight[0]))]
    for weight_li in doc_weight:
        for index, weight in enumerate(weight_li):
            doc_square_val[index].append(weight**2)
    
    dot_prod = []
    for index, ds_val_li in enumerate(doc_square_val):
        doc_prod = []
        for qs_val, ds_val in zip(query_square_val, ds_val_li):
            doc_prod.append(qs_val*ds_val)
        dot_prod.append(sum(doc_prod))

    return dot_prod

def cosinus_similarity(dot_product, query_distance, doc_distance):
    cosinus_sim = []
    for dot_val, doc_dval in zip(dot_product, doc_distance):
        cosinus_sim.append(dot_val/(query_distance*doc_dval))
    return cosinus_sim
    
def rank_similarity(cosinus_similarity):
    rank_dict = {}
    for index, sim_val in enumerate(cosinus_similarity):
        rank_dict[index] = sim_val
    print(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
    
#dataset (example)
id_language = "indonesian"
find_query = ['sistem']
id_language_test = 'Sistem adalah kumpulan elemen. Adalah kumpulan elemen yang saling berinteraksi. Sistem berinteraksi untuk mencapai tujuan.'

#pre processing
pre_procesing_result = pre_procesing(id_language_test, id_language)#udah bener
word_list_result = term_list(pre_procesing_result)#udah bener

#TF-IDF
tf_result = tf(pre_procesing_result, word_list_result)#udah bener
df_result = df(tf_result)#udah bener
ddf_result = ddf(pre_procesing_result, df_result)#udah bener
idf_result = idf(ddf_result) #udah bener
query_weight_result = query_weight(word_list_result, find_query, idf_result, tf_result) #udah bener
doc_weight_result = doc_weight(tf_result, idf_result) #udah bener

#PERHITUNGAN JARAK
query_distance_result = query_distance(query_weight_result)
document_distance_result = document_distance(doc_weight_result)

#DOT PRODUCT
dot_product_result = dot_product(query_weight_result, doc_weight_result)

#cosin similarity
cosinus_similarity_result = cosinus_similarity(dot_product_result, query_distance_result, document_distance_result)

#similarity_rank
rank_similarity(cosinus_similarity_result)