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

train_upper = int(len(lines) * 0.8)

random.shuffle(lines)

train = lines[:train_upper]
dev = lines[train_upper:]

train_file = open('data/cleaned_data/train.txt', 'w')
for i in train:
    train_file.write(i + '\n')
train_file.close()

dev_input = open('data/cleaned_data/dev_input.txt', 'w')
dev_answer = open('data/cleaned_data/dev_answer.txt', 'w')
for i in dev:
    val = random.randrange(1, len(i) - 1)
    dev_input.write(i[:val]+ '\n')
    dev_answer.write(i[val] + '\n')
    print(str(i))
    print(str(i[:val]))
    print(str(i[val]))
dev_input.close()
dev_answer.close()



