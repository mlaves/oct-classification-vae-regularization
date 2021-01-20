#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import metrics

results_vae = np.load('./results_test_vae.npy')
results_resnet = np.load('./results_test_resnet.npy')


def argmaxer(x):
    r = np.zeros_like(x)
    r[np.argmax(x)] = 1.0
    return r


for i, v in enumerate(results_vae[:,1]):
    results_vae[i, 1] = argmaxer(v)
    
for i, v in enumerate(results_resnet[:,1]):
    results_resnet[i, 1] = argmaxer(v)

print(metrics.classification_report(results_vae[:, 0], results_vae[:, 1], digits=3))
print('vae acc', metrics.accuracy_score(results_vae[:, 0], results_vae[:, 1]))

print(metrics.classification_report(results_resnet[:, 0], results_resnet[:, 1], digits=3))
print('resnet acc', metrics.accuracy_score(results_resnet[:, 0], results_resnet[:, 1]))
