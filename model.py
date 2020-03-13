from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from autoviml.Auto_ViML import Auto_ViML


def tpotClassifier(train_data, target_value):
    classifier = TPOTClassifier()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_data[target_value],
                                                        train_size=0.75, test_size=0.25)
    classifier.fit(X_train, y_train)
    score: float = classifier.score(X_test, y_test)
    classifier.export('my_pipeline.py')
    return classifier, score


def tpotRegressor(train_data, target_value):
    regressor = TPOTRegressor()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_data[target_value],
                                                        train_size=0.75, test_size=0.25)
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    regressor.export('my_pipeline.py')
    return regressor, score


def autoviml(train_data, target_value, reducer=True, verbosity=0):
    X_train, X_test, y_train, y_test = train_test_split(train_data, target_value,
                                                        train_size=0.75, test_size=0.25)
    viml = Auto_ViML(X_train, target_value, X_test,
                     hyper_param='GS', verbose=verbosity, feature_reduction=reducer)
    score = accuracy_score(y_train, y_test)
    return viml, score
