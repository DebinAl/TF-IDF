import string
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def pre_procesing(document, language_option = None):
    porter_stemmer = PorterStemmer()
    nltk_stopwords = set(stopwords.words(language_option))
    stemmer = StemmerFactory().create_stemmer() 
    document = str(document).casefold()                       #case folding    
    document = [sentence for sentence in document.split('.')] #split text

    for index, sentence in enumerate(document):
        print(sentence)
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
        inverse_doc_freq.append(math.log10(val)+1)
    return inverse_doc_freq

def doc_weight(tf, idf):
    doc_weight = []
    for doc_term_val, idf_val in zip(tf, idf):
        doc_weight.append([weight*idf_val for weight in doc_term_val])
    return doc_weight

def total_weight(doc_result):
    total = [0 for _ in range(3)]
    for doc in doc_result:
        for index, val in enumerate(doc):
            print(index, total[index], val )
            total[index] += val            
    return total
            
id_language = "indonesian"
test_query = ['sandang','pangan','papan']
id_language_test = 'Manusia memerlukan sandang. Manusia membutuhkan pangan pangan untuk hidup papan. Manusia memerlukan sandang untuk hidup, karena papan adalah rumah.'
id_text = "Studi tentang penggunaan teknologi dalam pendidikan menunjukkan pertumbuhan pesat dalam pengembangan platform pembelajaran online. Penelitian ini menggunakan corpus yang terdiri dari data dari berbagai institusi pendidikan. Analisis terhadap korpus ini mengungkapkan pola-pola penggunaan teknologi yang beragam dan strategi pengajaran yang efektif. Dengan memanfaatkan corpus ini, pendidik dapat merancang pengalaman pembelajaran yang lebih adaptif dan terkini bagi para siswa."

en_language = "english"
en_text = "Python programmers often tend like programming in python because it's like english. We call people who program in python pythonistas. This research utilizes a corpus consisting of data from various educational institutions. Analysis of this corpus reveals diverse patterns of technology use and effective teaching strategies. By leveraging this corpus, educators can design more adaptive and up-to-date learning experiences for students."


pre_procesing_result = pre_procesing(id_language_test, id_language)
tf_result = tf(pre_procesing_result, test_query)
df_result = df(tf_result)
ddf_result = ddf(pre_procesing_result, df_result)
idf_result = idf(ddf_result)
doc_result = doc_weight(tf_result, idf_result)
total_result =  total_weight(doc_result)

print(total_result)