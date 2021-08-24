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

embeddedTrain = np.zeros((metadataTrain.shape[0], 128))
embeddedTest = np.zeros((metadataTest.shape[0], 128))

for i, m in enumerate(metadataTrain):
    img = load_image(m.image_path())
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embeddedTrain[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

for i, m in enumerate(metadataTest):
    img = load_image(m.image_path())
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embeddedTest[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    print(distance(embeddedTrain[idx1], embeddedTrain[idx2]))
'''load cho một hs
for i in range(10,20):
    for j in range(i + 1, 20):
        show_pair(i, j)'''

'''tạo 2 mảng khoảng cách và gán nhán tương ứng với nhau'''

distances = []  # squared L2 distance between pairs
identical = []  # 1 if same identity, 0 otherwise

num = len(metadataTrain)
numTest = len(metadataTest)

for i in range(num - 1):
    for j in range(i + 1, num):
        distances.append(distance(embeddedTrain[i], embeddedTrain[j]))
        identical.append(1 if metadataTrain[i].name == metadataTrain[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)


thresholds = np.arange(0, 1, 0.01)

acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

'''# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score');
plt.plot(thresholds, acc_scores, label='Accuracy');
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
plt.xlabel('Distance threshold')
plt.legend()
plt.show()'''

'''
# vẽ đồ thị về positive và negative
dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

dist_pos = pd.cut(dist_pos, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]).value_counts()
dist_pos = dist_pos.tolist()
dist_pos = np.array(dist_pos)

dist_neg = pd.cut(dist_neg, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]).value_counts()
dist_neg = dist_neg.tolist()
dist_neg = np.array(dist_neg)
#print(dist_neg)
dist_neg = (dist_neg/10)#.astype(int)

plt.figure(figsize=(20,4))
trucx = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
plt.plot( trucx, dist_pos,label='Positive');
plt.plot(trucx, dist_neg, label='Negative');
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (pos. pairs)')
plt.legend()
plt.show()'''

targetsTrain = np.array([m.name for m in metadataTrain])
targetsTest = np.array([m.name for m in metadataTest])

encoderTrain = LabelEncoder()
encoderTrain.fit(targetsTrain)
# Numerical encoding of identities
ytrain = encoderTrain.transform(targetsTrain)

encoderTest = LabelEncoder()
encoderTest.fit(targetsTest)
# Numerical encoding of identities
ytest = encoderTest.transform(targetsTest)


train_idx = np.arange(metadataTrain.shape[0])
test_idx = np.arange(metadataTest.shape[0])


X_train = embeddedTrain[train_idx]
print(X_train.shape)
exit()
X_test = embeddedTest[test_idx]

y_train = ytrain[train_idx]
y_test = ytest[test_idx]


# vẽ ma trận hiệp phương sai
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train)
'''
acc_knn = accuracy_score(y_test, knn.predict(X_test))
print(f'KNN accuracy = {acc_knn}')
plot_confusion_matrix(knn, X_test, y_test)
plt.show()'''


import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

#phân biệt một cá nhân cụ thể
example_idx = 0
#dòng này load ảnh dựa trên ID được cung cấp
example_image = load_image(metadataTest[test_idx][example_idx].image_path())
#dòng này dự đoán bằng knn dựa trên đầu vào
# trả về là một số tự nhiên từ 1-33 tương đương với index của từng học sinh
example_prediction = knn.predict([X_test[example_idx]])

#dòng này xem id đó match với cái tên nào và in ra dự đoán
example_identity = encoderTest.inverse_transform(example_prediction)[0]
# các bước phía trên đã tìm tới điểm ngắn nhất của đầu vào
#bây giờ cần điểm ngắn nhất nhưng vẫn phải tính khoảng cách
distances, indices = knn.kneighbors([X_test[example_idx]])
print(distances[0][0])
plt.imshow(example_image)
plt.title(f'Recognized as {example_identity}')
if(distances[0][0] > 0.4):
    plt.title('UNKNOWN')
plt.show()

distanceNum = []
predictNUM = []
unknown = []

def show_predictions(countNUM):
    for idx in range(countNUM):
        example_image = load_image(metadataTest[test_idx][idx].image_path())
        example_prediction = knn.predict([X_test[idx]])
        distances, indices = knn.kneighbors([X_test[idx]])
        distanceNum.append(distances[0][0])
        #predictNUM.append(example_prediction[0])
        predictNUM.append(example_prediction[0] if distances[0][0]< 0.45 else 34)
        #example_identity = encoderTrain.inverse_transform(example_prediction)[0]
        #print(example_prediction[0])
        #print(ytest[idx])

show_predictions(numTest)

'''dist_num = pd.cut(distanceNum, [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]).value_counts()
dist_num = dist_num.tolist()
dist_num = np.array(dist_num)
print(dist_num)
'''
'''# vẽ convolution matrix dựa vào dòng predict.append
cm = confusion_matrix(y_test, predictNUM)
sns.heatmap'''