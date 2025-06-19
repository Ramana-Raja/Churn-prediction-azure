import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

df = pd.read_parquet("C:\Users\\raman\\PycharmProjects\\azure-mlops\\data.parquet")

mlflow.set_tracking_uri("sqlite:///C:\\Users\\raman\\PycharmProjects\\azure-mlops\\mlflow.db")
mlflow.set_experiment("my-experiment-1")
x = df.drop("Churn",axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [2, 4, 6, 8],
    'n_estimators': [2,5,10, 50, 100],
    'min_samples_split': [2,3,4],
    'criterion': ["gini", "entropy", "log_loss"],
    'min_samples_leaf' : [3,4,5],
}

for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        model = RandomForestClassifier(**params,random_state=42)
        model.fit(X_train,y_train)

        pred = model.predict(X_test)
        accuracy  = accuracy_score(y_test,pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Run with params: {params}")
        print(f"Accuracy: {accuracy:.4f}")

print(model.score(X_test,y_test))