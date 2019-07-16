from processor import Processor
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import jieba
import jieba.posseg as pseg
import utils
import numpy as np
import re


class Annotator(object):
    def __init__(self, config, form, path):
        self.conf = config
        self.form = form
        self.path = path
        self.processor = Processor(self.conf, self.path, self.form)
        self.sentences = self.processor.load()
        self.pattern, self.construction = self.processor.construct(4)
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
            feature[str(count)]["tag"] = ''
            feature[str(count)]["regex"] = 1
            feature[str(count)]["policy"] = 0
            feature[str(count)]["deriv"] = 1
            feature[str(count)]["agree"] = 0

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
    def agree(self):
        elements = self.form.split('+')
        shared = set([element for element in elements if elements.count(element) > 1])
        return shared

    # Third Layer
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

    def _observe(self, word, tag, count, sentence, feature):
        """
        Observe the series when the construction **do not** contain X or Y
        :param word: string
        :param tag: string
        :param count: int
        :param sentence: string
        :return: the type of the word
        """
        # phrase = sentence[count:count + self._length]
        phrase = self.form

        if word in self.construction.keys():
            return "constant"

        if tag in self.construction.keys():
            for constant in self._constants:
                if constant in phrase:
                    return "variable"
        else:
            if feature["regex"] != 1:
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
            else:
                if word in self.construction.keys():
                    return "constant"
                else:
                    return "variable"

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
            if len(segments) > 0 and words.index(word) >= sentence.index(segments[0]):
                step = self._judge(segments, word, sentence)
            else:
                step = self._observe(word, tag, count, sentence, feature[str(count)])

            score += self.conf.policies[step]
            # update the feature
            feature[str(count)]["tag"] = step
            feature[str(count)]["policy"] = score
            if score < 0 and feature[str(count)]["regex"] == 1.5:
                feature[str(count)]["regex"] = 0.5
            shared = self.agree()

            if word in shared:
                feature[str(count)]["agree"] = 1

            if step == "variable":
                if "X" in shared:
                    feature[str(count)]["agree"] = 1
                elif tag in shared:
                    feature[str(count)]["agree"] = 1

            count += 1

        return feature

    def _process(self):
        """
        Process the sentences
        :return: an array-like pairs
        """
        curves = list()

        # Update the features by regex and posseg
        for index, sentence in tqdm(self.sentences, desc="Processing the sentences"):
            if index not in self.features.keys():
                self.features[index] = self._construct(sentence)

            # Processed in first layer
            feature_regex = self._build(index, sentence)
            self.features[index].update(feature_regex)

            # Processed in third layer
            feature_policy = self._policy(index, sentence)
            agreements = [value["value"] for key, value in feature_policy.items() if value["agree"] == 1]
            not_agree = [item for item in agreements if agreements.count(item) == 1]
            for word in not_agree:
                for key, value in feature_policy.items():
                    if value["value"] == word:
                        feature_policy[key]["tag"] = "others"
                        feature_policy[key]["agree"] = 0
                        feature_policy[key]["regex"] = 1
            self.features[index].update(feature_policy)

        # Get the points
        for key, value in self.features.items():
            points = list()

            for position, features in value.items():
                point_x = int(position)
                point_y = features["policy"] * features["regex"] * features["deriv"]

                points.append((point_x, point_y))

            curves.append((key, points))

        return curves

    def fit(self):
        """ Fit the points of sentences """
        formulas, arguments, temp = list(), list(), list()
        curves = self._process()

        print("Start the fit the curve and get the candidate")
        for mark, points in curves:
            # Get the data of points
            x, y = utils.expend(points)
            ploy_reg = utils.PolynomialRegression(100)
            # Fit the points
            ploy_reg.fit(x, y)
            formulas.append((mark, ploy_reg))
            arguments.append((mark, x))
            temp.append((mark, y))

        return formulas, arguments, temp

    def transform(self):
        """ Bestow weights on candidate by derivation """
        formulas, arguments, temp = self.fit()

        print("Get the candidate by derivation")
        for mark, formula in formulas[4:5]:
            feature = self.features[mark]

            # Derivation
            x = utils.get(arguments, mark)
            y = utils.get(temp, mark)

            y_hat = formula.predict(x)

            sections = utils.growth(list(y_hat))

            candidates, phrase = [], ""

            for section in sections:
                for index in section:
                    phrase += feature[str(index)]['value']
                candidates.append(phrase)
                phrase = ""

            candidates = "".join(utils.cut(candidates, self._constants))

            for index, sentence in feature.items():
                if sentence["value"] in candidates:
                    sentence["deriv"] = 1.2 if sentence["policy"] > 0 else 0.2

            self.features[mark].update(feature)
        print("done")

    # Fourth Layer
    def cluster(self):
        self.transform()
        # Get the points of data
        curves = list()

        # Get the points
        for key, value in self.features.items():
            points = list()

            for position, features in value.items():
                point_x = int(position)
                point_y = features["policy"] * features["regex"] * features["deriv"]

                points.append((point_x, point_y))

            curves.append((key, points))

        # Clustering
        sentences = list()
        gmm = GaussianMixture(n_components=3)
        for mark, points in tqdm(curves, desc="clustering"):
            sentence = list()
            feature = self.features[mark]

            points = np.array(points)
            labels = gmm.fit_predict(points)

            for i in range(len(labels)):
                sentence.append((feature[str(i)], labels[i]))

            sentences.append(sentence)
        return sentences

    # Fifth Layer
    @staticmethod
    def find(sentence):
        classes = dict()
        for word, label in sentence:
            if str(label) not in classes.keys():
                classes[str(label)] = dict()

            if word["tag"] not in classes[str(label)].keys():
                classes[str(label)][word["tag"]] = 0

            classes[str(label)][word["tag"]] += 1

        classes = utils.classify(classes)
        return classes

    def annotate(self):
        sentences = self.cluster()

        print("Annotating...")
        results = []
        for sentence in sentences:
            classes = self.find(sentence)

            series = []
            for word, label in sentence:
                label = str(label)

                if classes[label] == "others":
                    if word["tag"] == "constant":
                        if word["regex"] != 1 and word["deriv"] != 1:
                            series.append((word["value"], "constant"))
                        else:
                            series.append((word["value"], "others"))
                    elif word["tag"] == "variable":
                        if word["regex"] != 1 or word["deriv"] != 1:
                            series.append((word["value"], "variable"))
                        else:
                            series.append((word["value"], "others"))
                    else:
                        if word["regex"] != 1:
                            series.append((word["value"], "variable"))
                        else:
                            series.append((word["value"], "others"))

                if classes[label] == "constant":
                    if word["tag"] == "constant":
                        if word["regex"] != 1 and word["deriv"] != 1:
                            series.append((word["value"], "constant"))
                        else:
                            series.append((word["value"], "others"))
                    elif word["tag"] == "variable":
                        if word["regex"] != 1 and word["deriv"] != 1:
                            series.append((word["value"], "variable"))
                        else:
                            series.append((word["value"], "others"))
                    else:
                        series.append((word["value"], "others"))

                if classes[label] == "variable":
                    if word["tag"] == "constant":
                        series.append((word["value"], "constant"))
                    else:
                        if word["regex"] == 1 and word["deriv"] == 1:
                            series.append((word["value"], "others"))
                        else:
                            series.append((word["value"], "variable"))

            for i in range(1, len(series) - 1):
                if series[i - 1][1] == "others" and series[i + 1][1] == "others":
                    series[i] = (series[i][0], "others")

            results.append(series)

        print("Write the data into the output file")
        # Store the data
        data = []
        for sentence in results:
            context, construction, content = "", [], []
            for word, label in sentence:
                if label == "others":
                    if len(construction) > 0:
                        content.append((construction, "cxn"))
                    construction = []
                    context += word
                else:
                    if len(context) > 0:
                        content.append((context, "context"))
                    context = ""
                    construction.append((word, label))

                if sentence.index((word, label)) == len(sentence) - 1:
                    if len(construction) > 0:
                        content.append((construction, "cxn"))
                    if len(context) > 0:
                        content.append((context, "context"))
            data.append(content)

        return data

    def store(self):
        data = self.annotate()

        with open(self.conf.output_path.format(self.form + "_" + self.path), "w") as out:
            # Write the metadata
            out.write('<?xml version="1.0" encoding="UTF-8"?>' + "\n")
            # Write the root tag
            out.write("<document>" + "\n")
            # Write the data
            for sentence in tqdm(data, desc="store the data"):
                content = "\t<sentence>"
                sentence = utils.reshape(sentence)

                for phrase, label in sentence:
                    # Pre-judgment
                    if label == "cxn":
                        temp = utils.tuple_to_str(phrase)

                        for constant in self._constants:
                            if constant not in temp:
                                label = "context"
                                phrase = temp
                                break

                    if label == "context":
                        content += phrase
                    else:
                        content += "<cxn>"
                        for words, tag in phrase:
                            words = jieba.cut(words)

                            for word in words:
                                if tag == "variable":
                                    content += "<variable>" + word + "</variable>"
                                else:
                                    content += "<constant>" + word + "</constant>"
                        content += "</cxn>"
                content += "</sentence>\n"
                out.write(content)

            out.write("</document>")
        out.close()

        print("Complete! The data was stored in" + self.conf.output_path.format(self.form + "_" + self.path))



