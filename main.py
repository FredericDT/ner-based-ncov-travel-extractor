from deeppavlov import configs, build_model
import pkuseg
import requests

import logging
import sys
import csv

KEY = '<AMAP API KEY>'

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
    ['B-GPE'],
    ["B-ORG"],
    # ['B-PERSON'],
]


def extract_pos(data):
    a = seg.cut(data)
    a = ner_model(a)
    a = list(filter(lambda x: x[1] in FILTER, zip(a[0], a[1])))
    return a


api_cache = {}


def loc_from_addr(addr):
    if addr in api_cache:
        return api_cache[addr]
    r = requests.get('https://restapi.amap.com/v3/geocode/geo?key={key}&address={addr}'.format(key=KEY, addr=addr))
    api_cache[addr] = r.json()['geocodes'][0]['location'] if len(r.json()['geocodes']) else None
    return api_cache[addr]


if __name__ == '__main__':
    i = 1
    k = 1
    with open('data.txt', 'r') as f:
        with open('visualize.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['id', 'loc'])
            for line in f.readlines():
                a = extract_pos(line)
                a = [loc_from_addr(j[0]) for j in a]
                logger.info('Result {} : {}'.format(i, a))
                for m in a:
                    if m is not None:
                        writer.writerow([k, m])
                        k += 1
                i += 1
