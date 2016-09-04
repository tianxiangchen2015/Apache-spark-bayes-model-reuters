from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os
import re
import os.path
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ["PYSPARK_PYTHON"]="/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7"
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.classification import NaiveBayesModel
import sys
reload(sys)
sys.setdefaultencoding('utf8')

cachedStopWords = stopwords.words("english")

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words
                  if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens);
    return filtered_tokens

from goose import Goose

if __name__ == "__main__":
    # Extract cleaned body text from the URL

    url = 'http://www.reuters.com/article/global-oil-idUSL3N16408T'
    g = Goose()
    article = g.extract(url=url)
    a = article.cleaned_text
    html_dict = []

    #Tokenize the text

    tokenhtml = tokenize(a)
    print(tokenhtml)

    # Put the text into a list of dict [{"text": body}]

    for i in range(0,len(tokenhtml)):
        body = ''
        body += tokenhtml[i]+' '
    html_dict.append({"text":body})

    # Load the data into spark

    sc = SparkContext()
    htmldata = sc.parallelize(html_dict)

    # Calculate TF-IDF

    tf = HashingTF().transform(htmldata.map(lambda doc: doc["text"], preservesPartitioning=True))
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)

    # Load the Naive Bayes model and predict

    sameModel = NaiveBayesModel.load(sc, "/Users/apple/Dropbox/2016Spring/COSC526/MacHW1/mymodel")
    predictionAndLabel = tfidf.map(lambda p: (sameModel.predict(p)))

    # Check the result

    print(format(predictionAndLabel.take(1)))