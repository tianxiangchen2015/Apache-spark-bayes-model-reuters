# Apache Spark --- Train bayes model with reuters

Naive Bayes classification is a simple probability classifier in machine learning algorithm. The Naive Bayes Classifier technique is based on the Bayesian theorem. Basically, naive Bayes is a conditional probability model. Given a class variable  and a dependent feature vector  through , Bayes’ theorem states the following relationship:
                        
                        p(y│x_1,….,x_n )=(p(y)p(x_1,…x_n |y))/(p(x_1,….x_n))

## Introduction

In this assignment, we will train a Naïve Bayes classification using “Reuters-21578” dataset. It is currently the most widely used test collection for text categorization research. The data was originally collected by Reuters and is already been labeled. The datasets contains 22 .sgm documents and each files contains approximately 1000 papers in various topics. In this assignment we will only focus on the following topics: “money, fx, crude, grain, trade, interest, wheat, ship, corn, oil, dlr, gas, oilseed, supply, sugar, gnp, coffee, veg, gold, soybean, bop, livestock, cpi.”.  We will parse these XML files and get the papers of interested topics, and create TF-IDF dictionary to train the Naïve Bayes. 

### Prerequisities

Required softwares:
```
[Apache Spark](http://spark.apache.org)
[Nature Language Toolkit](http://www.nltk.org)
[goose extractor](https://pypi.python.org/pypi/goose-extractor/)
```

### Dataset
```
[Reuters-21578](http://www.daviddlewis.com/resources/testcollections/reuters21578/)
```

## Authors

* **Tianxiang Chen (ORNL Research Assistant)** - [Linkedin HomePage](https://www.linkedin.com/in/tianxiang-chen-946543114?trk=nav_responsive_tab_profile)


## Acknowledgments

* Use sample code from https://chimpler.wordpress.com/2014/06/11/classifiying-documents-using-naive-bayes-on-apache-spark-mllib/
* 
    
