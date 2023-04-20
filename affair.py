import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("Affairs.csv")
target = "affairs"
data[target] = data[target].apply(lambda x: 0 if x == 0 else 1)
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_features", StandardScaler(), ["age", "yearsmarried", "religiousness", "education", "rating"]),
        ("ord_features", OrdinalEncoder(), ["gender", "children"]),
        ("nom_features", OneHotEncoder(handle_unknown="ignore"), ["occupation"])
    ]
)

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_depth": [None, 5, 10],
    "classifier__class_weight": ["balanced", "balanced_subsample", {0: 1, 1: 5}, {0: 1, 1: 10}],
}

grid = GridSearchCV(cls, param_grid=param_grid, cv=5, verbose=1, scoring="f1", n_jobs=-1)
grid.fit(x_train, y_train)
y_predict = grid.predict(x_test)
print(grid.best_score_)
print(grid.best_params_)
print(classification_report(y_test, y_predict))
