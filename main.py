#!/usr/bin/env python
"""
This script implements a full ML pipeline, from data sourcing to prediction, to
persistence of the experiment results. Make use of a utility file to hide
minutia of the implementation.
"""
import utils
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import constants as c
from reporting import make_report, pretty_print_report

np.random.seed(c.RANDOM_STATE)


def main():
    print("-------------------- Fetch Data")

    data = utils.fetch_data()
    X = data.drop(c.DROP_COLS + [c.TARGET_LABEL], axis=1)
    y = data[c.TARGET_LABEL]


    print("-------------------- Transform & Fit")

    # Setup feature transformations
    numerical_features = ['age', 'fare', 'sibsp']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('poly_maker', PolynomialFeatures(include_bias=False)),
        ('PCA', PCA())
    ], memory=c.CACHE_DIR)

    # Categoricals
    categorical_features = ['sex', 'pclass', 'embarked', 'parch']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ], memory=c.CACHE_DIR)

    # Make the preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    # Compose preprocessor and classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_jobs=-1))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    grid_searcher = GridSearchCV(
        estimator=pipeline,
        param_grid=dict(
            preprocessor__numerical__poly_maker__degree=[2, 3],
            preprocessor__numerical__imputer__strategy=['median', 'mean'],
            preprocessor__numerical__PCA__n_components=[3, 5, 7],
            classifier__n_estimators=[10, 100, 250],
            classifier=[RandomForestClassifier(n_jobs=-1), GradientBoostingClassifier()],
            classifier__random_state=[c.RANDOM_STATE],
        ),
        cv=3,
        refit=True,
        n_jobs=2,
    )
    grid_searcher.fit(X_train, y_train)

    print("-------------------- Evaluate Model")

    y_hat = grid_searcher.predict(X_test)[:, np.newaxis]
    y_probs = grid_searcher.predict_proba(X_test)[:, 1]

    report = make_report(y_hat, y_probs, y_test)
    print(pretty_print_report(report))

    experiment_info = {
        'random_state': c.RANDOM_STATE,
        'model': grid_searcher,
        'data': data,
        'report': report,
    }

    utils.persist_experiment(experiment_info)
    
if __name__ == '__main__':
    main()
