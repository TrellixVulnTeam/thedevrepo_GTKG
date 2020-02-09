'''
https://www.hackerrank.com/challenges/predicting-house-prices/problem
'''
from sklearn.linear_model import LinearRegression

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

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
for p in predictions:
    print(round(p,2))
