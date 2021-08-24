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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

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


metadataTrain = load_metadata('face_datasetTrain')
metadataTest = load_metadata('face_datasetTest')
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

embedded = np.zeros((metadataTrain.shape[0], 128))

for i, m in enumerate(metadataTrain):
    img = load_image(m.image_path())
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    #plt.figure(figsize=(8,3))
    #plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    print(distance(embedded[idx1], embedded[idx2]))
    #plt.subplot(121)
    #cv2.imshow("anh1",load_image(metadataTrain[idx1].image_path()))
    #plt.subplot(122)
    #cv2.imshow("anh2",load_image(metadataTrain[idx2].image_path()));
    #cv2.waitKey()

'''load cho một hs
for i in range(10,20):
    for j in range(i + 1, 20):
        show_pair(i, j)'''

'''tạo 2 mảng khoảng cách và gán nhán tương ứng với nhau'''

distances = []  # squared L2 distance between pairs
identical = []  # 1 if same identity, 0 otherwise

num = len(metadataTrain)
'''k = 0
for i in range(num):
    df = pd.DataFrame({'stt': [k], 'name': [metadataTrain[i].name]})
    df.to_csv('test.csv', mode='a', header=False)
    if(metadataTrain[i].name != metadataTrain[i+1].name):
        k = k+1'''


df2 = pd.read_csv('test.csv', date_parser={'x': eval, 'y': eval})


k=0
for i in range(num - 1):
    for j in range(i + 1, num):
        distances.append(k if distance(embedded[i], embedded[j]) < 0.4 else 0)
        identical.append(k if metadataTrain[i].name == metadataTrain[j].name else 0)
    if(metadataTrain[i].name != metadataTrain[i+1].name and i < num - 1):
        k = k + 1
distances = np.array(distances)
identical = np.array(identical)
cnf_matrix = confusion_matrix(distances, identical)

cm_df = pd.DataFrame(cnf_matrix,
                     index = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'],
                     columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'])
#Plotting the confusion matrix

sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

