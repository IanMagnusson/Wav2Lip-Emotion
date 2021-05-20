#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse
parser = argparse.ArgumentParser(description = "gets avg LSE from linsep file of video scores")
parser.add_argument('--infile', type=str, help='')
opt = parser.parse_args()

with open(opt.infile) as fin:
    lines = fin.readlines()

lse_d_sum, lse_c_sum = 0.0, 0.0
for line in lines:
    lse_d, lse_c = line.split()
    lse_d_sum += float(lse_d)
    lse_c_sum += float(lse_c)
print('lse_d:', lse_d_sum / len(lines))
print('lse_c:', lse_c_sum / len(lines))