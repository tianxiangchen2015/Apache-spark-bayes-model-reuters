from __future__ import print_function
import pprint
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import sgmllib
import os.path
import fnmatch
import sgmllib
import tarfile
from collections import Counter
os.environ["SPARK_HOME"] = "/usr/local/spark"
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')


def _not_in_sphinx():
    return '__file__' in globals()

class ReutersParser(sgmllib.SGMLParser):
    """Use sgmllib package to parse a SGML file and
        yield documents one at a time.
    """
    def __init__(self, verbose=0):
        sgmllib.SGMLParser.__init__(self, verbose)
        self._reset()

    def _reset(self):
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk)
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_topics:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.docs.append((self.topics, self.body))
        self._reset()

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.topics.append(self.topic_d)
        self.topic_d = ""

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0





def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));

    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens);
    return filtered_tokens

def filter_topics(docs):
    """
    Reads all of the documents and creates a new list of two-tuples
    that contain a single "label" and the body. Also, extract interested topics
    and replace topics with the number of index of this topic in categories list.
    For example, ('money', 'body') is ('0','body').
    """
    categories = ['money','fx', 'crude','grain',
                  'trade', 'interest', 'wheat', 'ship', 'corn', 'oil', 'dlr',
                  'gas', 'oilseed', 'supply', 'sugar', 'gnp', 'coffee', 'veg',
                  'gold', 'soybean','bop','livestock', 'cpi',
                  'money-fx','money-supply','veg-oil']
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "" or d[1]=="":
            continue
        for n in d[0]:
            if n in categories:
                d_tup = (categories.index(n), d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs

if __name__ == "__main__":
    # Open the Reuters21578 data set and create the parser
    filename = ["reuters21578/reut2-%03d.sgm" % r for r in range(0, 22)]
    parser = ReutersParser()
    pp = pprint.PrettyPrinter(indent=4)
    '''
    Parse the document and put all generated docs into
    a list.
    '''
    docs = []
    for fn in filename:
        for d in parser.parse(open(fn,'rb')):
            docs.append(d)
    print(docs[0])

    # Filt all the documents
    docs3=filter_topics(docs)

    length = len(docs3)
    pp.pprint(docs3[0])
    print(length)

    topic_list = []
    body_list = []
    tokendocs = []
    '''
    Tokenize all the bodys in the list and yield a list of tuple
    which contains (topic,body).
    '''
    for i in range(0,length):
        body = ''
        body_list= tokenize(docs3[i][1])
        for r in range(0,len(body_list)):
            body += body_list[r]+' '
        topic = str(docs3[i][0])
        tokendocs.append(topic+','+body)

    '''
    Create training data set by writing the clean data into
    a text file
    '''
    with open("training_data_final.txt",'w') as f:
        for s in tokendocs:
            f.write(s + '\n')



