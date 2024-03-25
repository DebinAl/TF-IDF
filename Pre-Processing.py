import string
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
        
en_language = "english"
en_text = "Python programmers often tend like programming in python because it's like english. We call people who program in python pythonistas. This research utilizes a corpus consisting of data from various educational institutions. Analysis of this corpus reveals diverse patterns of technology use and effective teaching strategies. By leveraging this corpus, educators can design more adaptive and up-to-date learning experiences for students."

id_language = "indonesian"
id_text = "Studi tentang penggunaan teknologi dalam pendidikan menunjukkan pertumbuhan pesat dalam pengembangan platform pembelajaran online. Penelitian ini menggunakan corpus yang terdiri dari data dari berbagai institusi pendidikan. Analisis terhadap korpus ini mengungkapkan pola-pola penggunaan teknologi yang beragam dan strategi pengajaran yang efektif. Dengan memanfaatkan corpus ini, pendidik dapat merancang pengalaman pembelajaran yang lebih adaptif dan terkini bagi para siswa."

print(pre_procesing(en_text, en_language))
# print(pre_procesing(id_text, id_language))