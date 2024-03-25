import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def pre_procesing(document, language_option = None):
    nltk_stopwords = set(stopwords.words(language))
    stemmer = StemmerFactory().create_stemmer() 
    
    #case folding
    document = str(document).casefold() 
    
    #tokenisasi
    document = [sentence for sentence in document.split('.')]
    print(document)
    
    for index, sentence in enumerate(document):
        arr_sentence = []
        #punctuation
        sentence = str(sentence).translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        #punctuation
        for word in sentence.split():
            #stemming
            word = stemmer.stem(word)
            #stopword
            if word not in nltk_stopwords:
                arr_sentence.append(word)
                
        document[index] = " ".join(arr_sentence)
        
    return document
        
# language = 'english'
language = 'indonesian'
text = 'Dalam era digital ini, teknologi, , $ %       # " , informasi telah menjadi bagian tak terpisahkan dari kehidupan sehari-hari. Hal ini membawa dampak yang signifikan bagi berbagai aspek kehidupan manusia, termasuk pendidikan, ekonomi, dan sosial.'

paragraph = pre_procesing(text, language)
print(paragraph)