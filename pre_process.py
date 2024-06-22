import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet') 

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')

def pre_process(text):

    lentext = len(text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = [word.lower() for word in text.split()]

    text = [word for word in text if word not in stop_words]
    
    text = [lemma.lemmatize(word) for word in text]

    text = [word for word in text if len(word) > 1]

    results = " ".join(text)
    
    return results + ' ' + str(lentext)