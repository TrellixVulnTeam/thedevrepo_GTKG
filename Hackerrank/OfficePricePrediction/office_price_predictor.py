'''
https://www.hackerrank.com/challenges/predicting-office-space-price/problem
'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

F, H = map(int, input().split(' '))
X_train = [] ; y_train=[]

for _ in range(0,H):
    tmp = list(map(float, input().split(' ')))
    X_train.append(tmp[:F])
    y_train.append(tmp[F])

T = int(input())
X_test = []
for _ in range(0,T):
    tmp = list(map(float, input().split(' ')))
    X_test.append(tmp)

poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)

clf = LinearRegression()
clf.fit(X_poly_train, y_train)

predictions = clf.predict(X_poly_test)
for p in predictions:
    print(round(p,2))