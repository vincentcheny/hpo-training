#!/bin/bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
apt-get install unzip
unzip uncased_L-12_H-768_A-12.zip
echo uncased_L-12_H-768_A-12.zip is ready. Please manually download train.csv from https://www.kaggle.com/sergeykalutsky/introducing-bert-with-tensorflow