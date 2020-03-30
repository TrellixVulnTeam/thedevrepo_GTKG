from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape, y_train.shape)
logistic_classifier = LogisticRegression()
svm_classifier = SVC()
svm_prob_classifier = SVC(probability=True)
rf_classifier = RandomForestClassifier()

hard_voting_classifier = VotingClassifier(estimators = [('logreg',logistic_classifier), ('svc', svm_classifier), ('rf', rf_classifier)])
soft_voting_classifier = VotingClassifier([('logreg',logistic_classifier), ('svc_p', svm_prob_classifier), ('rf', rf_classifier)], voting='soft')

hard_voting_classifier.fit(X_train,y_train)
soft_voting_classifier.fit(X_train,y_train)

for clf in [hard_voting_classifier, soft_voting_classifier]:
    y_pred = clf.predict(X_test)
    print("Classifier: {}, Accuracy: {}".format(clf.__class__.__name__, accuracy_score(y_pred, y_test)))

