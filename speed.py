import numpy as np
import os.path
from model import create_model
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
import seaborn as sns
import time
class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.png':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]


metadataTrain = load_metadata('test31')
embeddedTrain = np.zeros((metadataTrain.shape[0], 128))

for i, m in enumerate(metadataTrain):
    img = load_image(m.image_path())
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embeddedTrain[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
num = len(metadataTrain)
'''
metadataTest = load_metadata('face_datasetTest')
embeddedTest = np.zeros((metadataTest.shape[0], 128))
for i, m in enumerate(metadataTest):
    img = load_image(m.image_path())
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embeddedTest[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
numTest = len(metadataTest)
   '''

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    print(distance(embeddedTrain[idx1], embeddedTrain[idx2]))

distances = []  # squared L2 distance between pairs
identical = []  # 1 if same identity, 0 otherwise


for i in range(num - 1):
    for j in range(i + 1, num):
        distances.append(distance(embeddedTrain[i], embeddedTrain[j]))
        identical.append(1 if metadataTrain[i].name == metadataTrain[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)

targetsTrain = np.array([m.name for m in metadataTrain])
encoderTrain = LabelEncoder()
encoderTrain.fit(targetsTrain)
ytrain = encoderTrain.transform(targetsTrain)
train_idx = np.arange(metadataTrain.shape[0])
X_train = embeddedTrain[train_idx]
y_train = ytrain[train_idx]

'''targetsTest = np.array([m.name for m in metadataTest])
encoderTest = LabelEncoder()
encoderTest.fit(targetsTest)
ytest = encoderTest.transform(targetsTest)
test_idx = np.arange(metadataTest.shape[0])
X_test = embeddedTest[test_idx]
y_test = ytest[test_idx]'''


knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train)


import warnings

# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')
img = load_image("speed/20.png")
img = (img / 255.).astype(np.float32)
embeddedTest = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


start_time = time.time()
example_prediction = knn.predict([embeddedTest])
print(example_prediction)
print("--- %.5f seconds ---" % (time.time() - start_time))