from processor import Processor
from tqdm import tqdm
from scipy.misc import derivative
import jieba
import jieba.posseg as pseg
import utils
import math
import numpy as np
import re


class Annotator(object):
    def __init__(self, config, form, path):
        self.conf = config
        self.form = form
        self.path = path
        self.processor = Processor(self.conf, self.path, self.form)
        self.sentences = self.processor.load()
        self.pattern, self.construction = self.processor.construct(3)
        self.features = dict()
        self._length = len(self.form.split('+'))
        self._constants = [key for key, value in self.construction.items() if value == "constant"]

    # Initialize
    def initialize(self):
        jieba.load_userdict(self.conf.userdict)

    # First Layer
    @staticmethod
    def _construct(sentence):
        """ Build the preliminary data structure for sentence """
        count = 0
        feature = dict()

        for word in sentence:
            if str(count) not in feature.keys():
                feature[str(count)] = dict()

            feature[str(count)]["value"] = word
            feature[str(count)]["regex"] = 1
            feature[str(count)]["policy"] = 0
            feature[str(count)]["deriv"] = 1

            count += 1

        return feature

    def _match(self, sentence):
        """ Get the candidate by RegEx preliminarily """
        return re.findall(self.pattern, sentence)

    def _build(self, index, sentence):
        """
        Bestow weights on candidate by regex
        :param index: string - the mark of the sentence
        :param sentence: string
        :return: dict - update the policy of the feature
        """
        feature = self.features[index]
        constructions = self._match(sentence)

        for construction in constructions:
            start = sentence.index(construction)
            end = start + len(construction)

            for key, val in feature.items():
                if int(key) not in range(start, end):
                    continue

                val["regex"] += 0.5

        return feature

    # Second Layer
    @staticmethod
    def _posseg(sentence):
        """
        Word Segmentation and POS Tagging
        :param sentence: string
        :return: tuple - words and their tags
        """
        words, tags = list(), list()
        pairs = [pseg.cut(word) for word in sentence]

        for pair in pairs:
            temp = dict(pair)
            words += list(temp.keys())
            tags += list(temp.values())

        return words, tags

    def _observe(self, word, tag, count, sentence):
        """
        Observe the series when the construction **do not** contain X or Y
        :param word: string
        :param tag: string
        :param count: int
        :param sentence: string
        :return: the type of the word
        """
        phrase = sentence[count:count + self._length]

        if word in self.construction.keys():
            return "constant"

        if tag in self.construction.keys():
            for constant in self._constants:
                if constant in phrase:
                    return "variable"

        return "others"

    def _judge(self, segments, word, sentence):
        """
        Observe the series when the construction **do** contain X or Y
        :param segments: an array of clauses which contains X, Y or Z
        :param word: string
        :param sentence: string
        :return: the type of the word
        """
        for segment in segments:
            if word not in segment:
                continue

            if sentence.index(word) in range(sentence.index(segment), sentence.index(segment) + len(segment)):
                if word in self.construction.keys():
                    return "constant"
                else:
                    return "variable"
            else:
                continue

        return "others"

    def _complex(self, sentence):
        """
        Check if the construction contains X, Y or Z
        :param sentence: string
        :return: an array of clauses which contains X, Y or Z
        """
        segments = list()

        if utils.contains(["X", "Y", "Z"], self.construction):
            clauses = re.split(r'\W+', sentence)

            for clause in clauses:
                if utils.includes(clause, self._constants):
                    segments.append(clause)

        return segments

    def _policy(self, index, sentence):
        """
        Create policy based on pos of word
        :param index: string - the mark of the sentence
        :param sentence: string
        :return: dict - update the policy of the feature
        """
        feature = self.features[index]
        words, tags = self._posseg(sentence)
        score, count = 0, 0
        segments = self._complex(sentence)

        for word, tag in zip(words, tags):
            if len(segments) > 0:
                step = self._judge(segments, word, sentence)
            else:
                step = self._observe(word, tag, count, sentence)

            score += self.conf.policies[step]
            # update the feature
            feature[str(count)]["policy"] = score

            count += 1

        return feature

    def _process(self):
        """ Process the sentences """
        curves = list()

        # Update the features by regex and posseg
        for index, sentence in tqdm(self.sentences, desc="Processing the sentences"):
            if index not in self.features.keys():
                self.features[index] = self._construct(sentence)

            # Processed in first layer
            feature_regex = self._build(index, sentence)
            self.features[index].update(feature_regex)

            # Processed in second layer
            feature_policy = self._policy(index, sentence)
            self.features[index].update(feature_policy)

        # Get the points
        for key, value in self.features.items():
            points = list()

            for position, features in value.items():
                point_x = int(position)
                point_y = features["policy"] * features["regex"]

                points.append((point_x, point_y))

            curves.append((key, points))

        return curves

    def fit(self):
        """ Fit the points of sentences """
        formulas, arguments = list(), list()
        curves = self._process()

        print("Start the fit the curve and get the candidate")
        for mark, points in curves:
            x, y = utils.expend(points)
            curve = np.polyfit(x, y, math.ceil(len(points) / self._length))
            formula = np.poly1d(curve)
            formulas.append((mark, formula))
            arguments.append((mark, x))

        return formulas, arguments

    def transform(self):
        """ Bestow weights on candidate by derivation """
        formulas, arguments = self.fit()

        print("Get the candidate by derivation")
        for mark, formula in formulas[1:2]:
            feature = self.features[mark]

            # Derivation
            variables = utils.get(arguments, mark)
            values = list()

            if len(variables):
                for variable in variables:
                    values.append((variable, derivative(formula, variable, dx=1e-6)))

            # TODO: Check the status of fitting
