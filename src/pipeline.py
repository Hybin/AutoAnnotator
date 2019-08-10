from xml.etree import ElementTree
from tqdm import tqdm
import re


class Pipeline(object):
    def __init__(self, config, form, path):
        self.conf = config
        self.form = form
        self.path = path

    # Simple Matching for pipeline annotation
    def _load(self):
        tree = ElementTree.parse(self.conf.input_path.format(self.path))
        root = tree.getroot()
        sentences = root.findall('sentence')

        return sentences

    def _get_pattern(self):
        components = self.form.split('+')

        construction = ''
        for component in tqdm(components, desc='Construction Representation'):
            if re.search("[a-zA-Z]", component):
                construction += '.{1,10}?'
            else:
                construction += component

        return construction

    def _match(self):
        sentences = self._load()
        pattern = self._get_pattern()

        paragraphs = list()
        for sentence in sentences:
            text = sentence.text
            construction = re.findall(pattern, text)

            paragraph = list()
            if len(construction):
                position = []
                for entry in construction:
                    if text.find(entry) != 0 and (0, 'context') not in position:
                        position.append((0, 'context'))

                    position += [(text.find(entry), 'cxn'), (text.find(entry) + len(entry), 'context')]

                if (len(text), 'context') not in position:
                    position.append((len(text), 'context'))

                ranges = list()
                for i in range(len(position) - 1):
                    ranges.append([position[i], position[i + 1]])

                for r in ranges:
                    paragraph.append((text[r[0][0]:r[1][0]], r[0][1]))
            else:
                paragraph.append((text, 'context'))
            paragraphs.append(paragraph)

        contents = list()
        for paragraph in paragraphs:
            sentence = list()
            for phrase, flag in paragraph:
                if flag == 'cxn':
                    words = []
                    for i in range(len(phrase)):
                        if phrase[i] in pattern:
                            words.append((phrase[i], 'constant'))
                        else:
                            words.append((phrase[i], 'variable'))
                    sentence.append((words, flag))
                else:
                    sentence.append((phrase, flag))
            contents.append(sentence)

        return contents

    @staticmethod
    def _create_node(name, content):
        node = ElementTree.Element(name)
        node.text = content
        return node

    def annotate(self):
        contents = self._match()

        leaves = list()
        for sentence in tqdm(contents):
            parent = self._create_node('sentence', '')
            for phrase, flag in sentence:
                if flag == 'context':
                    node = self._create_node('', phrase)
                    parent.append(node)
                else:
                    cxn = self._create_node('cxn', '')
                    for element, tag in phrase:
                        node = self._create_node(tag, element)
                        cxn.append(node)
                    parent.append(cxn)
            leaf = re.sub('</?>', '', ElementTree.tostring(parent, encoding='utf-8').decode())
            leaves.append(leaf)

        with open(self.conf.output_pipe.format(self.path), 'w') as fp:
            fp.write('<?xml version="1.0" encoding="UTF-8"?>' + '\n')
            fp.write('<document>' + '\n')

            for leaf in leaves:
                fp.write('\t' + leaf + '\n')

            fp.write('</document>')
