import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def pre_procesing(document, language_option = None):
    nltk_stopwords = set(stopwords.words(language))
    stemmer = StemmerFactory().create_stemmer() 
    #case folding
    document = str(document).casefold() 
    print(document)
    #case folding
    document = [str(term).translate(str.maketrans(' ',' ',string.punctuation)) for term in str(document).split('.')]
    print(document)
    #remove white space
    document = [" ".join(word.split()) for word in document]
    print(document)
    #stopword
    document = [stopword_clean for sentence in document for stopword_clean in sentence.split() if stopword_clean not in nltk_stopwords]
    print(document)
    #stemming
    document = [stemmer.stem(stem_word) for sentence in document for stem_word in sentence]
    
# language = 'english'
language = 'indonesian'
text = 'Dalam era digital ini, teknologi informasi telah menjadi bagian tak terpisahkan dari kehidupan sehari-hari. Internet, smartphone, dan berbagai aplikasi telah mengubah cara kita berinteraksi, bekerja, dan bahkan belajar. Hal ini membawa dampak yang signifikan bagi berbagai aspek kehidupan manusia, termasuk pendidikan, ekonomi, dan sosial. Dengan kemajuan teknologi, koneksi antar individu semakin mudah terjalin, memungkinkan pertukaran informasi secara cepat dan global.'
pre_procesing(text, language)