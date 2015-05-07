import numpy as np
index2char = {}
char2index = {}
index2ans = {}
a = 0
chrmap = open('data/48_idx_chr.map','r')
for line in chrmap:
    line = line.split()
    index2char[int(line[1])] = line[0]
    char2index[line[0]] = int(line[1])
    index2ans[int(line[1])] = line[2]
PHONES = 48
FBANKS = 69

def read_fbank(filename):
    fin = open("data/fbank/" + filename +".ark", "r")
    lines = []
    for line in fin:
        line = line.rstrip().split(' ')
        line[0] = line[0].split('_')
        line[0][2] = int(line[0][2])
        line[1:] = [float(ll) for ll in line[1:]]
        lines += [line]
    return lines

def read_label(filename):
    fin = open("data/label/" + filename + ".lab", "r")
    lines = []
    for line in fin:
        line = line.rstrip().split(',')
        line[0] = line[0].split('_')
        line[0][2] = int(line[0][2])
        line[1] = char2index[line[1]]
        lines += [line]
    return lines

def read_weight(filename):
    import json
    f = open("model/" + filename, "r")
    return json.loads(f.readline())

