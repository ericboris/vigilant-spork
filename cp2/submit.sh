#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Eric Boris,eboris91\nAllyson Ely,elyall17" > submit/team.txt

# train model and save the trained model to work directory work
python3 src/rnn.py train --work_dir work

# make predictions on example data submit it in pred.txt
python3 src/rnn.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# submit data
cp -r data submit/data

# make zip file
zip -r submit.zip submit
