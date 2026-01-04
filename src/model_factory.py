from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # or MultinomialNB if text data


def get_model(cfg):
    name = cfg["name"]

    if name == "logistic":
        return LogisticRegression(**cfg.get("logistic", {}))

    elif name == "naive_bayes":
        return GaussianNB(**cfg.get("naive_bayes", {}))

    elif name == "decision_tree":
        return DecisionTreeClassifier(**cfg.get("decision_tree", {}))

    elif name == "random_forest":
        return RandomForestClassifier(**cfg.get("random_forest", {}))

    else:
        raise ValueError(f"Unknown model: {name}")
