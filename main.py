#!/usr/bin/env python
import utils
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

random_state = 42
np.random.seed(random_state)


TARGET_LABEL = 'target'

if __name__ == '__main__':
    print("-------------------- Fetch Data")

    X, y = utils.fetch_data()

    print("-------------------- Transform Data")

    # Perform some feature transformation, like polynomialize
    X = utils.transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("-------------------- Train Model")

    clf = RandomizedSearchCV(
        estimator=RandomForestClassifier(n_jobs=2),
        param_distributions=dict(
            n_estimators=[10, 100, 250],
            max_depth=[3, 7, 10],
        ),
    )
    clf.fit(X_train, y_train)

    print("-------------------- Evaluate Model")

    y_hat = clf.predict(X_test)[:, np.newaxis]
    y_probs = clf.predict_proba(X_test)[:, 1]

    report = utils.make_report(y_hat, y_probs, y_test)
    print(utils.pretty_print_report(report))

    print("-------------------- Memoize Experiment")
    # persist with joblib
