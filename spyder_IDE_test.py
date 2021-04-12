# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons
from matplotlib.image import imread
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

image = imread('island.png')


X = image.reshape(-1, 4)
kmeans = KMeans(n_clusters=3).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

plt.imshow(segmented_img)

b = kmeans.cluster_centers_[[1,2,0,1,1]]
