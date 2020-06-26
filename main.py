from deeppavlov import configs, build_model
import pkuseg

import logging
import sys

logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
seg = pkuseg.pkuseg()
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
FILTER = [
    ['B-GPE'], ["B-ORG"], ['B-PERSON'], ['B-PERSON']
]


def extract_pos(data):
    a = seg.cut(data)
    a = ner_model(a)
    a = list(filter(lambda x: x[1] in FILTER, zip(a[0], a[1])))
    return a


if __name__ == '__main__':
    i = 1
    with open('data.txt', 'r') as f:
        for line in f.readlines():
            a = extract_pos(line)
            logger.info('Result {} : {}'.format(i, a))
            i += 1
