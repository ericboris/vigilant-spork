#!/usr/bin/env bash
set -e
set -v
python src/rnn.py test --work_dir work --test_data $1 --test_output $2
