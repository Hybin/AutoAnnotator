from processor import Processor
from utils import *
from tqdm import tqdm
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

            count += 1

        return feature

    def _match(self, sentence):
        """ Get the candidate by RegEx preliminarily """
        return re.findall(self.pattern, sentence)

    def build(self, sentence):
        """ Bestow weights on candidate by regex"""
        feature = self._construct(sentence)
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
    # TODO: Bestow weights on candidate by policy
