import json
import os
import random

os.mkdir('data/cleaned_data')


# parse all of the xcopa-master/data files
sub_dir = os.listdir('data/xcopa-master/data')

lines = []
for dir in sub_dir:
    test = 'test.' + dir + '.jsonl'
    val = 'val.' + dir + '.jsonl'
    with open('data/xcopa-master/data/' + dir + '/' + test) as doc:
        for j in doc.readlines():
            dict = json.loads(j)
            lines.append(dict['premise'])
            lines.append(dict['choice1'])
            lines.append(dict['choice2'])

    with open('data/xcopa-master/data/' + dir + '/' + val) as doc:
        for j in doc.readlines():
            dict = json.loads(j)
            lines.append(dict['premise'])
            lines.append(dict['choice1'])
            lines.append(dict['choice2'])

with open('data/xcopa-master/data-gmt/et/test.et.jsonl') as doc:
    for j in doc.readlines():
        dict = json.loads(j)
        lines.append(dict['premise'])
        lines.append(dict['choice1'])
        lines.append(dict['choice2'])

with open('data/xcopa-master/data-gmt/et/val.et.jsonl') as doc:
    for j in doc.readlines():
        dict = json.loads(j)
        lines.append(dict['premise'])
        lines.append(dict['choice1'])
        lines.append(dict['choice2'])

train_upper = int(len(lines) * 0.6)
dev_upper = train_upper + int((len(lines) * 0.2))

random.shuffle(lines)

train = lines[:train_upper]
dev = lines[train_upper:dev_upper]
test = lines[dev_upper:]

train_file = open('data/cleaned_data/train.txt', 'w')
for i in train:
    train_file.write(i + '\n')
train_file.close()

dev_file = open('data/cleaned_data/dev.txt', 'w')
for i in dev:
    dev_file.write(i + '\n')
dev_file.close()

test_file = open('data/cleaned_data/test.txt', 'w')
for i in test:
    test_file.write(i + '\n')
test_file.close()


