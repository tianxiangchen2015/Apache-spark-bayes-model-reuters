from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
import os.path
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ["PYSPARK_PYTHON"]="/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7"
from pyspark import SparkContext

def parseLine(line):
    dataset = []
    parts = line.split(',')
    labels = float(parts[0])
    features = parts[1]
    dataset.append({"label":labels,"text":features})
    return dataset

# Read the data and parse it into a list
sc = SparkContext("local[4]","NaiveBayesClassifier")
data = sc.textFile("training_test_data.txt").map(parseLine)

'''
Split data into labels and features, transform
preservesPartitioning is not really required
since map without partitioner shouldn't trigger repartitiong
'''
# Extract all the "labels"
labels = data.map(lambda doc: doc[0]["label"], preservesPartitioning = True)

for x in labels.take(3):
    print x
# Perform TF-IDF
tf = HashingTF().transform(data.map(lambda doc: doc[0]["text"], preservesPartitioning=True))
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

# Combine lables and tfidf and create LabeledPoint data
dataset = labels.zip(tfidf).map(lambda x: LabeledPoint(x[0], x[1]))

for x in dataset.take(3):
    print(x)
result=[]
'''
Random split dataset - 60% as training data and 40% as testing.
Train and test the model 10 times. Then put the accuracy into result[]
'''
for i in range(0,10):
    training, test = dataset.randomSplit([0.6,0.4],seed=i)
    model = NaiveBayes.train(training,1.0)
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
    result.append(accuracy)
    print(accuracy)
print(result)
# Save the model
model.save(sc,"mynewmodel")