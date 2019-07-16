import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot
import re


def expend(points):
    """
    Expend the data structure of point
    :param points: an array of tuples like [(x, y), ... ]
    :return: np-array of x and y respectively
    """
    x, y = list(), list()
    for first, second in points:
        x.append(first)
        y.append(second)

    return np.array(x).reshape(-1, 1), np.array(y)


def contains(targets, construction):
    """
    Check if the construction contains X, Y or Z
    :param targets: an array of variable like [X, Y, Z]
    :param construction: a dict contains the formal information of the construction
    :return: boolean True or False
    """
    for target in targets:
        if target in construction.keys():
            return True
        else:
            continue

    return False


def includes(clause, constants):
    """
    Check if the clause catches all constants
    :param clause: a string
    :param constants: a array of constants of construction
    :return: boolean True or False
    """
    for constant in constants:
        if constant not in clause:
            return False

    return True


def get(pairs, mark):
    """
    Get the second value by first value in pairs
    :param pairs: tuple - [(x, y), ... ]
    :param mark: string
    :return: the second value whose first value is mark
    """
    for first, second in pairs:
        if first == mark:
            return second
        else:
            continue

    return ''


def PolynomialRegression(degree):
    """
    Fit the data by the pipeline [PolynomialFeatures, StandardScaler, LinearRegression]
    :param degree: int
    :return: the model of fitting the data
    """
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])


def growth(values):
    """
    Get the increasing section
    :param values: array. y_hat that predicted by model
    :return: array. increasing sections
    """
    section, sections = list(), list()
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            if len(section) > 0:
                sections.append(section)
            section = []
        else:
            if i - 1 not in section:
                section.append(i - 1)

            if i not in section:
                section.append(i)

    return sections


def cut(phrases, constants):
    """
    cut the phrase that do not contains constant of
    construction
    :param phrases: list. list of words.
    :param constants: list. list of constants of construction
    :return: list. list of candidates of construction
    """
    candidates = []

    for phrase in phrases:
        segments = re.split(r'\W+', phrase)

        for segment in segments:
            if len(segment) < len(constants):
                continue
            else:
                for constant in constants:
                    if constant not in segment:
                        continue
                    else:
                        candidates.append(segment)
                        break

    return candidates


def maximum(dictionary):
    max_score = max(dictionary.values())

    tag = [key for key, val in dictionary.items() if val == max_score]

    return tag[0]


def classify(classes):
    """
    Map the class to the biggest score of label
    :param classes: dict. {label: {tag: score}, ...}
    :return: dict. {label: tag}
    """
    rule = {
        "others": "others",
        "variable": "variable",
        "constant": "constant"
    }

    dictionary = {}

    for label, tags in classes.items():
        dictionary[label] = rule[maximum(tags)]

    return dictionary


def cut_pairs(pairs):
    """
    cut the pairs by the second value
    :param pairs: list of pairs
    :return: list of pairs
    """
    whole, part = [], []

    for item, label in pairs:
        if len(part) == 0:
            part = [(item, label)]
        else:
            if part[0][1] == label:
                part.append((item, label))
            else:
                whole.append(part)
                part = [(item, label)]

        continue

    return whole


def reshape(sentence):
    results = []

    for text, label in sentence:
        construction = []
        if label == "context":
            results.append((text, label))
        else:
            variable, constant = "", ""

            count = 0
            for word, tag in text:
                if tag == "variable":
                    if len(constant) > 0:
                        construction.append((constant, "constant"))
                    constant = ""
                    variable += word
                else:
                    if len(variable) > 0:
                        construction.append((variable, "variable"))
                    variable = ""
                    constant += word

                # Check if it is the last word
                if count == len(text) - 1:
                    if len(constant) > 0:
                        construction.append((constant, "constant"))
                    if len(variable) > 0:
                        construction.append((variable, "variable"))
                count += 1

            results.append((construction, label))
    return results


def tuple_to_str(constructions):
    content, results = constructions, ""

    for word, label in content:
        results += word

    return results


def plot_model(model, x, y):
    y_hat = model.predict(x)

    plot.scatter(x, y)
    # plot.plot(x, y_hat, color="r")
    plot.xlabel('x')
    plot.ylabel('y')
    plot.legend(loc=4)
    plot.title('regression')
    plot.show()
