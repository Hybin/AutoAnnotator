import json


class Config(object):
    conf_file = '../config/config.json'

    def __init__(self):
        with open(Config.conf_file, encoding='utf-8') as cf:
            conf_dict = json.load(cf)

        for key, value in conf_dict.items():
            if isinstance(value, str) or isinstance(value, dict) or \
                    isinstance(value, int) or isinstance(value, float) or isinstance(value, list):
                setattr(self, key, value)
