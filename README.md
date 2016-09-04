# Apache Spark --- Train bayes model with reuters

Naive Bayes classification is a simple probability classifier in machine learning algorithm. The Naive Bayes Classifier technique is based on the Bayesian theorem. Basically, naive Bayes is a conditional probability model. Given a class variable  and a dependent feature vector  through , Bayes’ theorem states the following relationship:
                        
                        p(y│x_1,….,x_n )=(p(y)p(x_1,…x_n |y))/(p(x_1,….x_n))

## Introduction

In this assignment, we will train a Naïve Bayes classification using “Reuters-21578” dataset. It is currently the most widely used test collection for text categorization research. The data was originally collected by Reuters and is already been labeled. The datasets contains 22 .sgm documents and each files contains approximately 1000 papers in various topics. In this assignment we will only focus on the following topics: “money, fx, crude, grain, trade, interest, wheat, ship, corn, oil, dlr, gas, oilseed, supply, sugar, gnp, coffee, veg, gold, soybean, bop, livestock, cpi.”.  We will parse these XML files and get the papers of interested topics, and create TF-IDF dictionary to train the Naïve Bayes. 

## Prerequisities

Required softwares:
```
[Apache Spark](http://spark.apache.org)
[Nature Language Toolkit](http://www.nltk.org)
[goose extractor](https://pypi.python.org/pypi/goose-extractor/)
```

## Dataset
```
[Reuters-21578](http://www.daviddlewis.com/resources/testcollections/reuters21578/)
```
## Approch

1. Parse the XML documents using python build in “sgml” library, extract all the topics and body and put them in to a python list “docs= [(topic, body)]”. Then, filter the list and get only the interested topics. Also, eliminate all the dirty data, such as documents with an empty body
2. Tokenize and stem the body part using Nature Language Processing Toolkit (NLTK) so that we can only keep the meaningful content. In this part, we use Lancast Stemmer to stem all the words.
3. Vectorizing the documents and TF-IDF calculation. Load the documents into Spark Context, which is a resilient distributed dataset (RDD). RDD is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel. After load the data into Spark, implement a vector representation of the documents using TF-IDF method. Term frequency  is the number of times that term  appears in document . And IDF is the inverse document frequency
4. Training and testing Naïve bayes model. Randomly split all the data into two parts – 60% of data as training data and 40% as testing data. 

## Author

* **Tianxiang Chen (ORNL Research Assistant)** - [Linkedin HomePage](https://www.linkedin.com/in/tianxiang-chen-946543114?trk=nav_responsive_tab_profile)


## Acknowledgments

* Use sample code from https://chimpler.wordpress.com/2014/06/11/classifiying-documents-using-naive-bayes-on-apache-spark-mllib/
* 
    
