from xml.etree import ElementTree
from tqdm import tqdm
import re


class Processor(object):
    def __init__(self, config, path, form):
        self.conf = config
        self.path = self.conf.input_path.format(path)
        self.form = form

    def _parse(self):
        """ Parse the .xml file """
        tree = ElementTree.parse(self.path)
        root = tree.getroot()
        sentences = root.findall("sentence")

        return sentences

    def load(self):
        """ Extract the sentences from the raw material """
        sentences = list()

        # Get the sentence nodes by parsing the .xml file
        nodes = self._parse()

        for node in tqdm(nodes, desc="Loading the raw material"):
            sentences.append(node.text)

        return sentences

    def construct(self, window):
        """ Build the RegEx pattern for construction and
            Extract the formal information of the construction
        """
        pattern, construction = '', {}

        components = self.form.split("+")
        for component in tqdm(components, desc="Analyzing the construction"):
            if re.search("[a-zA-Z]", component):
                # Variable of Construction
                pattern += "[^\W.]{1," + str(window) + "}?"
                construction[component] = 'variable'
            else:
                # Constant of Construction
                pattern += component
                construction[component] = 'constant'

        return pattern, construction
