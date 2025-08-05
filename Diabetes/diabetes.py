import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


df = pd.read_csv("C:\\Users\\knox0\\Desktop\\diabetes.csv")
print(df.head())
print(df.info())
print(df.describe())



print(df.columns)

check_col =["Glucose","Insulin","BMI","SkinThickness","Pregnancies"]

for i in check_col:
    zero_count = (df[i] == 0).sum()
    print(zero_count)

X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=15)

check_col_fill=["Glucose","Insulin","BMI","SkinThickness","Pregnancies"]



medians = {}

for i in check_col_fill:
    value = X_train[X_train[i] != 0][i].median()
    medians[i] = value
    X_train[i] = X_train[i].replace(0,value)
  
for i in check_col_fill:
    X_test[i] = X_test[i].replace(0,medians[i])
    print(X_test)

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

ada = AdaBoostClassifier()

ada.fit(X_train_sc,y_train)
y_pred_ada = ada.predict(X_test_sc)

cls = classification_report(y_test, y_pred_ada)
acc = accuracy_score(y_test, y_pred_ada)
conf = confusion_matrix(y_test, y_pred_ada)

print("Confusion \n",cls)
print("accuracy \n", acc)
print("confusion \n", conf)

adaboost_param = {
        "n_estimators" : [50, 70, 100, 120, 150, 200],
        "learning_rate" : [0.001, 0.01, 0.1, 1, 10]
}

grid = GridSearchCV(estimator = AdaBoostClassifier(), param_grid = adaboost_param, cv = 5, verbose = 1, n_jobs = -1)
grid.fit(X_train, y_train)

print(grid.best_estimator_)

ada=AdaBoostClassifier(learning_rate=1,n_estimators=100)

ada.fit(X_train_sc,y_train)
y_pred = ada.predict(X_test_sc)


cls = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Confusion \n",cls)
print("accuracy \n", acc)
print("confusion \n", conf)
