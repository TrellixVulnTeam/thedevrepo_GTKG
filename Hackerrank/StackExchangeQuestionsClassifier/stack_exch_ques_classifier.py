'''
https://www.hackerrank.com/challenges/stack-exchange-question-classifier/problem
'''
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

path = 'Hackerrank/StackExchangeQuestionsClassifier/training.json'
training_json = []
with open(path, 'r') as f:
    for line in f:
        training_json.append(line)

N = int(training_json[0])
data = [] ; X = []; y = []

training_json = training_json[1:]

for i,instance in enumerate(training_json):
    data.append(json.loads(instance))
    
    X.append(data[i]['excerpt'])
    y.append(data[i]['topic'])

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

pipeline.fit(X,y)

# Runtime input
N = int(input())
X_test = []

for _ in range(N):
    obs = json.loads(input())
    X_test.append(obs['excerpt'])

predictions = pipeline.predict(X_test)
for p in predictions:
    print(p)
