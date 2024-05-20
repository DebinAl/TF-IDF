from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
import string
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

root = Tk()
root.geometry('+200+190') #window position
root.resizable(0,0)
root.title("Pointwise Grayscale")
vScale = IntVar()

window = Frame(root, padx=20, pady=20)
frameView = Frame(window, bg="#000000")
frameNavigation = Frame(window, pady=25)
frameNavigationButton = Frame(frameNavigation)

window.pack()

# Pre-processing TF-IDF + VSM
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

def term_list_vsm(pre_procesed):
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

def idf1(ddf):
    inverse_doc_freq = []
    for val in ddf:
        inverse_doc_freq.append(val+1)
    return inverse_doc_freq

# TF-IDF
def doc_weight_tfidf(tf, idf):
    doc_weight = []
    for doc_term_val, idf_val in zip(tf, idf):
        doc_weight.append([weight*idf_val for weight in doc_term_val])
    return doc_weight

# TF-IDF
def total_weight_tfidf(doc_result, doc_len):
    total = [0 for _ in range(doc_len)]
    for doc in doc_result:
        for index, val in enumerate(doc):
            total[index] += val            
    return total

# VSM
def query_weight_vsm(word_list, find_query, idf, tf):
    query_weight = []
    for word, idf_val in zip(word_list, idf):
        if word in find_query :
            val = 1*idf_val
            query_weight.append(val)
        else:
            val = 0*idf_val
            query_weight.append(val)
    
    return query_weight

# VSM
def doc_weight_vsm(tf, idf):
    weight_by_Term = []
    for term_freq, doc_freq in zip(tf, idf):
        term_row = []
        for term_val in term_freq:
            term_row.append(term_val*doc_freq)
        weight_by_Term.append(term_row)
    return weight_by_Term
    
# VSM
def query_distance_vsm(query_weight):
    square_val = []
    for weight in query_weight:
        square_val.append(weight**2) 
    return math.sqrt(sum(square_val)), square_val

def document_distance_vsm(doc_weight):
    square_val = [[] for _ in range(len(doc_weight[0]))]
    for weight_li in doc_weight:
        for index, weight in enumerate(weight_li):
            square_val[index].append(weight**2)
            
    doc_distance = []
    for val in square_val:
        doc_distance.append(math.sqrt(sum(val)))

    return doc_distance, square_val
    
    
def dot_product_vsm(query_weight, doc_weight):
    query_square_val = []
    for weight in query_weight:
        query_square_val.append(weight**2) 
    
    doc_square_val = [[] for _ in range(len(doc_weight[0]))]
    for weight_li in doc_weight:
        for index, weight in enumerate(weight_li):
            doc_square_val[index].append(weight**2)
    
    dot_prod = []
    dot_raw = []
    for index, ds_val_li in enumerate(doc_square_val):
        doc_prod = []
        for qs_val, ds_val in zip(query_square_val, ds_val_li):
            doc_prod.append(qs_val*ds_val)
        dot_raw.append(doc_prod)
        dot_prod.append(sum(doc_prod))

    return dot_prod, dot_raw

def cosinus_similarity_vsm(dot_product, query_distance, doc_distance):
    cosinus_sim = []
    for dot_val, doc_dval in zip(dot_product, doc_distance):
        cosinus_sim.append(dot_val/(query_distance*doc_dval))
    return cosinus_sim
    
def rank_similarity_vsm(cosinus_similarity):
    rank_dict = {}
    for index, sim_val in enumerate(cosinus_similarity):
        rank_dict[index] = sim_val
    return sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)



#  ============================================================
#  ====================== LUPAKAN SAJA ========================
#  ==================== Tkinter Activity ======================
#  ============================================================
def getQuery():
    for widget in frameView.winfo_children():
        widget.destroy()

    tf_result = []
    operation = navigation_label.cget("text")
    test_document = document_input.get("1.0", 'end-1c')

    test_query = query_input.get()
    test_query = str(test_query).translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    test_query = test_query.split()
    
    pre_procesing_result = pre_procesing(test_document, "indonesian")
        
    if (operation == "TFIDF"):
        tf_result = tf(pre_procesing_result, test_query)
    
    if (operation == "VSM"):
        term_list = term_list_vsm(pre_procesing_result)
        tf_result = tf(pre_procesing_result, term_list)
    
    df_result = df(tf_result)
    ddf_result = ddf(pre_procesing_result, df_result)
    idf_result = idf(ddf_result)
    
    if (operation == "TFIDF"):
        idf1_result = idf1(idf_result)
        doc_weight_result = doc_weight_tfidf(tf_result, idf1_result)
        total_result = total_weight_tfidf(doc_weight_result, len(pre_procesing_result))
        
        tfidftable(len(pre_procesing_result))
        for index, query in enumerate(test_query):
            e = Label(frameView, text=query, fg="#000000", font=('Times New Roman', 12))
            e.grid(column=0, row=index+2, sticky=EW, padx=1, pady=1)
        for index_i, i in enumerate(tf_result):
            f = Frame(frameView, background="#6e6e6e")
            f.grid(column=1, row=index_i+2, padx=1, pady=1, sticky=EW)
            for index_val, val in enumerate(i):
                f.columnconfigure(index_val, weight=1)
                sl = Label(f, text=val, fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=index_val, row=0, sticky=EW, padx=1, pady=1)
        for index, val in enumerate(df_result):
            e = Label(frameView, text=val, fg="#000000", font=('Times New Roman', 12))
            e.grid(column=2, row=index+2, sticky=EW, padx=1, pady=1)           
        for index, val in enumerate(ddf_result):
            e = Label(frameView, text=val, fg="#000000", font=('Times New Roman', 12))
            e.grid(column=3, row=index+2, sticky=EW, padx=1, pady=1)
        for index, val in enumerate(idf_result):
            e = Label(frameView, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12))
            e.grid(column=4, row=index+2, sticky=EW, padx=1, pady=1)
        for index, val in enumerate(idf1_result):
            e = Label(frameView, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12))
            e.grid(column=5, row=index+2, sticky=EW, padx=1, pady=1)
        for index_i, i in enumerate(doc_weight_result):
            f = Frame(frameView, background="#6e6e6e")
            f.grid(column=6, row=index_i+2, padx=1, pady=1, sticky=EW)
            for index_val, val in enumerate(i):
                f.columnconfigure(index_val, weight=1)
                sl = Label(f, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=index_val, row=0, sticky=EW, padx=1, pady=1)
        for index_i, val in enumerate(total_result):
            if index_i == 0:
                f = Frame(frameView, background="#000000")
                f.grid(column=6, row=index_i+3+len(doc_weight_result), padx=1, pady=1, sticky=EW)
                sl = Label(frameView, text="Nilai Bobot : ", fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=5, row=index_i+3+len(doc_weight_result), padx=1, pady=1, sticky=S)
                for i in range(len(pre_procesing_result)):
                    f.columnconfigure(i, weight=1)
                    sl = Label(f, text=("Doc "+ str(i)), fg="#000000", font=('Times New Roman', 12), padx=10)
                    sl.grid(column=i, row=0, padx=1, pady=1, sticky=EW)
            e = Label(f, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12))
            e.grid(column=index_i, row=1, sticky=EW, padx=1, pady=1)
        
    if (operation == "VSM"):
        vsmtable(len(pre_procesing_result))
        query_weight_result = query_weight_vsm(term_list, test_query, idf_result, tf_result)
        document_weight_result = doc_weight_vsm(tf_result, idf_result)
        query_distance_result, query_distance_raw = query_distance_vsm(query_weight_result)
        document_distance_result, document_distance_raw = document_distance_vsm(document_weight_result)
        dot_product_result, dot_product_raw = dot_product_vsm(query_weight_result, document_weight_result)
        cosinus_similarity_result = cosinus_similarity_vsm(dot_product_result, query_distance_result, document_distance_result)
        rank_similarity_result = rank_similarity_vsm(cosinus_similarity_result)
        for index, query in enumerate(term_list):
            e = Label(frameView, text=query, fg="#000000", font=('Times New Roman', 12))
            e.grid(column=0, row=index+2, sticky=EW, padx=1, pady=1)
        for index_i, i in enumerate(tf_result):
            f = Frame(frameView, background="#6e6e6e")
            f.grid(column=1, row=index_i+2, padx=1, pady=1, sticky=EW)
            f.columnconfigure(0, weight=1)
            if term_list[index_i] in set(test_query):
                sl = Label(f, text=1, fg="#000000", font=('Times New Roman', 12), padx=10)
            else:
                sl = Label(f, text=0, fg="#000000", font=('Times New Roman', 12), padx=10)
            sl.grid(column=0, row=0, sticky=EW, padx=1, pady=1)
            for index_val, val in enumerate(i):
                f.columnconfigure(index_val+1, weight=1)
                sl = Label(f, text=val, fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=index_val+1, row=0, sticky=EW, padx=1, pady=1)
        for index, val in enumerate(df_result):
            e = Label(frameView, text=val, fg="#000000", font=('Times New Roman', 12))
            e.grid(column=2, row=index+2, sticky=EW, padx=1, pady=1)  
        for index, val in enumerate(ddf_result):
            e = Label(frameView, text=val, fg="#000000", font=('Times New Roman', 12))
            e.grid(column=3, row=index+2, sticky=EW, padx=1, pady=1)
        for index, val in enumerate(idf_result):
            e = Label(frameView, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12))
            e.grid(column=4, row=index+2, sticky=EW, padx=1, pady=1)
        for (index_q, q), (index_i, i) in zip(enumerate(query_weight_result), enumerate(document_weight_result)):
            f = Frame(frameView, background="#6e6e6e")
            f.grid(column=5, row=index_i+2, padx=1, pady=1, sticky=EW)
            f.columnconfigure(0, weight=1)
            sl = Label(f, text=format(q, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
            sl.grid(column=0, row=0, sticky=EW, padx=1, pady=1)
            for index_val, val in enumerate(i):
                f.columnconfigure(index_val+1, weight=1)
                sl = Label(f, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=index_val+1, row=0, sticky=EW, padx=1, pady=1)
        for index_q, q in enumerate(query_distance_raw):
            f = Frame(frameView, background="#6e6e6e")
            f.grid(column=6, row=index_q+2, padx=1, pady=1, sticky=EW)
            f.columnconfigure(0, weight=1)
            sl = Label(f, text=format(q, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
            sl.grid(column=0, row=0, sticky=EW, padx=1, pady=1)
            for index_val, val in enumerate(document_distance_raw):
                f.columnconfigure(index_val+1, weight=1)
                sl = Label(f, text=format(val[index_q], ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=index_val+1, row=0, sticky=EW, padx=1, pady=1)
        for index, val in enumerate(document_distance_result):
            if index == 0:
                f = Frame(frameView, background="#6e6e6e")
                f.grid(column=6, row=len(query_distance_raw)+3, padx=1, pady=1, sticky=EW)
                f.columnconfigure(0, weight=1)
                sl = Label(f, text="sqrt(sum(Q)) & sqrt(sum(Di))", fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=0, row=0, columnspan= (len(document_distance_result)+1), sticky=EW, padx=1, pady=1)
                sl = Label(f, text=format(query_distance_result, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=0, row=1, sticky=EW, padx=1, pady=1)
            f.columnconfigure(index+1, weight=1)
            sl = Label(f, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
            sl.grid(column=index+1, row=1, sticky=EW, padx=1, pady=1)
        for index_q in range(len(query_distance_raw)):
            f = Frame(frameView, background="#6e6e6e")
            f.grid(column=7, row=index_q+2, padx=1, pady=1, sticky=EW)
            f.columnconfigure(0, weight=1)
            for index_val, val in enumerate(dot_product_raw):
                f.columnconfigure(index_val, weight=1)
                sl = Label(f, text=format(val[index_q], ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=index_val, row=0, sticky=EW, padx=1, pady=1)
        for index, val in enumerate(dot_product_result):
            if index == 0:
                f = Frame(frameView, background="#6e6e6e")
                f.grid(column=7, row=len(query_distance_raw)+3, padx=1, pady=1, sticky=EW)
                f.columnconfigure(0, weight=1)
                sl = Label(f, text="Sum(Q*Di)", fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=0, row=0, columnspan= (len(document_distance_result)+1), sticky=EW, padx=1, pady=1)
            f.columnconfigure(index, weight=1)
            sl = Label(f, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
            sl.grid(column=index, row=1, sticky=EW, padx=1, pady=1)
        for index_q, val in enumerate(cosinus_similarity_result):
            if(index_q == 0):
                f = Frame(frameView, background="#6e6e6e")
                f.grid(column=8, row=2, padx=1, pady=1, sticky=EW)
            f.columnconfigure(index_q, weight=1)
            sl = Label(f, text=format(val, ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
            sl.grid(column=index_q, row=0, sticky=EW, padx=1, pady=1)
        for index_q, val in enumerate(rank_similarity_result):
            if(index_q == 0):
                f = Frame(frameView, background="#6e6e6e")
                f.grid(column=8, row=3, padx=1, pady=1, rowspan=2, sticky=EW)
            sl = Label(f, text=("Rank "+str(val[0]+1)), fg="#000000", font=('Times New Roman', 12))
            sl.grid(column=index_q, row=0, sticky=EW, padx=1, pady=1)
            f.columnconfigure(index_q, weight=1)
            sl = Label(f, text=format(val[1], ".3f"), fg="#000000", font=('Times New Roman', 12), padx=10)
            sl.grid(column=val[0], row=1, sticky=EW, padx=1, pady=1)

        
def operation(val):
    if val == 1:
        navigation_label.config(text="TFIDF")
    if val == 2:
        navigation_label.config(text="VSM")

def tfidftable(doc_len):
    header = ['Query', 'tf', 'df', 'D/df', 'IDF', 'IDF+1', 'W = tf*(IDF+1)']
    for index, list in enumerate(header):
        e = Label(frameView, text=list, fg="#000000", font=('Times New Roman', 12))
        if(list == 'tf' or list == 'W = tf*(IDF+1)'):
            e.grid(column=index, row=0, sticky=EW, padx=1, pady=1)
            f = Frame(frameView, background="#000000")
            f.grid(column=index, row=1, padx=1, pady=1, sticky=EW)
            for i in range(doc_len):
                f.columnconfigure(i, weight=1)
                sl = Label(f, text=("Doc "+ str(i)), fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=i, row=0, padx=1, pady=1, sticky=EW)
        else:
            e.grid(column=index, row=0, rowspan=2, sticky=NSEW, padx=1, pady=1)

def vsmtable(doc_len):
    header = ['Query', 'tf', 'df', 'D/df', 'IDF', 'W = tf*IDF', 'Jarak', 'Dot Produk', 'Similarity']
    for index, list in enumerate(header):
        e = Label(frameView, text=list, fg="#000000", font=('Times New Roman', 12))
        if(list == 'tf' or list == 'W = tf*IDF' or list == 'Jarak' or list == 'Dot Produk' or list == 'Similarity'):
            e.grid(column=index, row=0, sticky=EW, padx=1, pady=1)
            f = Frame(frameView, background="#000000")
            f.grid(column=index, row=1, padx=1, pady=1, sticky=EW)
            for i in range(doc_len):
                f.columnconfigure(i, weight=1)
                if (list != 'Dot Produk' and list != 'Similarity'):
                    if (i == 0):
                        sl = Label(f, text="Query", fg="#000000", font=('Times New Roman', 12), padx=10)
                        sl.grid(column=i, row=0, padx=1, pady=1, sticky=EW)
                sl = Label(f, text=("Doc "+ str(i)), fg="#000000", font=('Times New Roman', 12), padx=10)
                sl.grid(column=i+1, row=0, padx=1, pady=1, sticky=EW)
        else:
            e.grid(column=index, row=0, rowspan=2, sticky=NSEW, padx=1, pady=1)


# Tkinter Setup
window.columnconfigure(0, minsize=500)
window.rowconfigure(0, minsize=100)

frameView.columnconfigure(0, minsize=100)
frameView.columnconfigure(1, minsize=75)
frameView.columnconfigure(2, minsize=75)
frameView.columnconfigure(3, minsize=75)
frameView.columnconfigure(4, minsize=75)
frameView.columnconfigure(5, minsize=75)
frameView.columnconfigure(6, minsize=75)
frameView.grid(row=0, column=0, sticky=NSEW)

navigation_label = Label(frameNavigationButton, text="Operation", fg="#000000", font=('Times New Roman', 12))
tf_idf_button = Button(frameNavigationButton, text="TF-IDF", width=20, command=lambda: operation(1))
vsm_button = Button(frameNavigationButton, text="VSM", width=20, command=lambda: operation(2))

document_label = Label(frameNavigation, text="Document", fg="#000000", font=('Times New Roman', 12))
document_input = Text(frameNavigation, width=10, height=5)

query_label = Label(frameNavigation, text="Find Query", fg="#000000", font=('Times New Roman', 12))
query_input = Entry(frameNavigation, width=10)
query_button = Button(frameNavigation, text="Enter", width=20, command=getQuery)

frameNavigation.columnconfigure(1, minsize=300)
frameNavigation.grid(row=1, column=0, sticky=NSEW)

navigation_label.grid(row=1, column=0, sticky=W, pady=5)
tf_idf_button.grid(row=2, column=0, sticky=EW)
vsm_button.grid(row=3, column=0, sticky=EW,pady = 3)
frameNavigationButton.grid(row=0,rowspan=9, column=0, sticky=NSEW, padx=10)

document_label.grid(row=0, column=1, sticky=W, padx=5, pady=5)
document_input.grid(row=1, column=1, sticky=EW, padx=5)
query_label.grid(row=3, column=1, sticky=W, padx=5, pady=5)
query_input.grid(row=4, column=1, sticky=EW, padx=5)
query_button.grid(row=5, column=1, sticky=EW, padx=5, pady=5)

window.mainloop()

#  ============================================================
#  ====================== LUPAKAN SAJA ========================
#  ==================== Tkinter Activity ======================
#  ============================================================