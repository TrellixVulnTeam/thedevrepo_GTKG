import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_excel(
    'C:/gsailesh/pyws/Experiments/FlightPricePrediction/data/Data_Train.xlsx')

print(train.head())
print(train.info())

train.dropna(inplace=True)

X, y = train.drop(['Price'], axis=1), train['Price']

X['Date_of_Journey'] = pd.to_datetime(X['Date_of_Journey'],
                                      infer_datetime_format=True, cache=True)

X['Day'] = X['Date_of_Journey'].dt.dayofweek.astype('int32')

X['Dep_Time'] = pd.to_datetime(
    X['Dep_Time'], infer_datetime_format=True, cache=True)
X['Arrival_Time'] = pd.to_datetime(
    X['Arrival_Time'], infer_datetime_format=True, cache=True)
X['Duration'] = X['Duration'].str.split('h', expand=True)[0]
X['Duration'] = X['Duration'].str.replace('[A-Za-z]', '').astype('int32')

num_cols = X.select_dtypes(exclude=['object']).columns
cat_cols = X.select_dtypes(include=['object']).columns

print(X.head())

lbl_encoder = LabelEncoder()
for col in cat_cols:
    X[col] = lbl_encoder.fit_transform(X[col])

feature_subset = X.drop(['Route'], axis=1).select_dtypes(
    include='int32').columns
print(feature_subset)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()

model = GradientBoostingRegressor()

param_grid = {'learning_rate': [0.1, 0.05], 'n_estimators': [100, 200, 300, 400, 500], 'criterion': [
    'friedman_mse'], 'min_samples_split': [2], 'min_samples_leaf': [1], 'max_depth': [2, 3, 4]}

grid_model = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, verbose=2)
regression_pipeline = Pipeline([('scaler', scaler), ('model', grid_model)])

regression_pipeline.fit(X_train[feature_subset], y_train)
y_pred = regression_pipeline.predict(X_val[feature_subset])

result = mean_squared_error(y_val, y_pred, squared=False)
print(result)
