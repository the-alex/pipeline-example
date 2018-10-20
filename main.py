#!/usr/bin/env python
"""
This script implements a full ML pipeline, from data sourcing to prediction, to
persistence of the experiment results. Make use of a utility file to hide
minutia of the implementation.
"""
import utils
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from constants import *

random_state = 42
np.random.seed(random_state)


def main():
    print("-------------------- Fetch Data")

    data = utils.fetch_data()
    
    X = data.drop(TARGET_LABEL, axis=1)
    y = data[TARGET_LABEL]

    print("-------------------- Transform Data")

    # Perform some feature transformation, like polynomialize
    X = utils.transform(X)
    X = X[FEATURES]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("-------------------- Train Model")

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)

    clf = GridSearchCV(
        estimator=RandomForestClassifier(n_jobs=-1),
        param_grid=dict(
            n_estimators=[10, 100, 250],
            max_depth=[3, 7, 10],
        ),
        refit=True,
        cv=cv
    )
    clf.fit(X_train, y_train)

    print("-------------------- Evaluate Model")

    y_hat = clf.predict(X_test)[:, np.newaxis]
    y_probs = clf.predict_proba(X_test)[:, 1]

    report = utils.make_report(y_hat, y_probs, y_test)
    print(utils.pretty_print_report(report))


if __name__ == '__main__':
    main()
