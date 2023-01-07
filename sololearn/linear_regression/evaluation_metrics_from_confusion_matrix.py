#!/usr/bin/env python

tp, fp, fn, tn = [int(x) for x in input().split()]

total = sum([tp, fp, fn, tn])

accuracy = (tp + tn) / total
precsison = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precsison * recall / (precsison + recall)

print(round(accuracy, 4))
print(round(precsison, 4))
print(round(recall, 4))
print(round(f1_score, 4))
