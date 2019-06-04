import numpy as np


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

    return np.array(x), np.array(y)


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
