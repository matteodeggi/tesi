import numpy as np
import scipy as sc
from scipy.stats import iqr
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def euclidean(s1, s2):
    return np.sqrt(np.sum((s1 - s2) ** 2))

'''
Calculates dynamic time warping Euclidean distance between two
sequences. Option to enforce locality constraint for window w.
'''
def DTWDistance(s1, s2, w=None):

    DTW = {}

    if w:
        w = max(w, abs(len(s1) - len(s2)))

        for i in range(-1, len(s1)):
            for j in range(-1, len(s2)):
                DTW[(i, j)] = float('inf')

    else:
        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        if w:
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = (s1[i] - s2[j]) ** 2
                DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
        else:
            for j in range(len(s2)):
                dist = (s1[i] - s2[j]) ** 2
                DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])



def accuracyED(x_train, x_test, y_train, y_test):
    cnt = 0.0
    for k, t in enumerate(x_test):
        d_min = 10000000
        y_class = None
        for j,i in enumerate(x_train):
            d = euclidean(t, i)
            if (d < d_min):
                d_min = d
                y_class = y_train[j]
        if y_class == y_test[k]:
            cnt += 1.0
    return cnt/float(len(y_test))

def accuracyDTW(x_train, x_test, y_train, y_test):
    cnt = 0.0
    for k, t in enumerate(x_test):
        d_min = 10000000
        y_class = None
        for j,i in enumerate(x_train):
            d = DTWDistance(t, i)
            if (d < d_min):
                d_min = d
                y_class = y_train[j]
        if y_class == y_test[k]:
            cnt += 1.0
    return cnt/float(len(y_test))


# METODI PER L'APPLICAZIONE DEGLI ALGORITMI DI CLASSIFICAZIONE E CALCOLO DELLA LORO ACCURACY

# Random Forest
def accuracyRandomForest(x_train, x_test, y_train, y_test):
    regressor = RandomForestClassifier(n_estimators=100, max_features='log2')#, random_state=0)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    return accuracy_score(y_test, y_pred, normalize=True)

# C4.5
def accuracyDecisionTree(x_train, x_test, y_train, y_test):
    clf = DecisionTreeClassifier().fit(x_train, y_train)
    dt_pred = clf.predict(x_test)
    return accuracy_score(y_test, dt_pred, normalize=True)

# K Nearest Neighbors
def accuracyKNN(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier().fit(x_train, y_train)
    kn_pred = knn.predict(x_test)
    return accuracy_score(y_test, kn_pred, normalize=True)

# SVC
def accuracySVC(x_train, x_test, y_train, y_test):
    svm = SVC(kernel='linear').fit(x_train, y_train)
    svm_pred = svm.predict(x_test)
    return accuracy_score(y_test, svm_pred, normalize=True)

# Naive Bayes
def accuracyNaiveBayes(x_train, x_test, y_train, y_test):
    gnb = GaussianNB().fit(x_train, y_train)
    gnb_pred = gnb.predict(x_test)
    return accuracy_score(y_test, gnb_pred, normalize=True)
    
# AdaBoost
def accuracyAdaBoost(x_train, x_test, y_train, y_test):
    abo = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, y_train)
    #AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)
    abo_pred = abo.predict(x_test)
    return accuracy_score(y_test, abo_pred, normalize=True)



# MAIN

#path = "C:/Users/Mattia/Desktop/Dati/3D/"
#path2 = "C:/Users/Mattia/Desktop/Dati/tirocinio/"
path2 = ""

NUM_FEATURES_TOT = 11
valori_features = []
classi = []
accuracy_finali = []
valori_ts = []

# CALCOLO FEATURES
 
with open(path2 + 'UrbanObservatory.csv', 'r', encoding='utf8') as dati:
#with open(path2 + 'UrbanObservatory.csv', 'r', encoding='utf8') as dati:
#with open(path + 'datastream_annotati2.csv', 'r', encoding='utf8') as dati:
    for row in dati:
        #riga = row.strip().split(';')
        riga = row.strip().split(',')
        #classe = int(riga[8])
        classe = int(riga[0])
        classi.append(classe)
        #valori = np.array(riga[9:]).astype(np.float)
        valori = np.array(riga[1:]).astype(np.float)
        valori_ts.append(valori)
        media = np.mean(valori)
        mediana = np.median(valori)
        maxim = np.max(valori)
        minim = np.min(valori)
        std_dev = np.std(valori)
        rms = np.sqrt(np.mean(np.square(valori)))
        quantile = np.quantile(valori, 0.4)
        i_q_r = iqr(valori)
        simmetria = skew(valori)
        curtosi = kurtosis(valori)
        rang = maxim - minim
        #features = [maxim, rang, std_dev, rms, media, curtosi, simmetria, mediana, minim, quantile, i_q_r]
        features = [rang, maxim, std_dev, rms, media, minim, quantile, mediana, curtosi, simmetria, i_q_r] #nuova analisi varianza 17/06
        valori_features.append(features)


# CALCOLO DELLE ACCURACY CON AUMENTO PROGRESSIVO DELLE FEATURES CONSIDERATE


shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, train_size=0.7)
Wi = np.array([tt for tt in valori_ts])
Yi = np.array([c for c in classi])

for i in range(0, NUM_FEATURES_TOT):

    Xi = np.array([f[0:i+1] for f in valori_features])
    
    for train_index, test_index in shuffle_split.split(Xi, Yi):
        X_train, X_test = Xi[train_index], Xi[test_index]
        Y_train, Y_test = Yi[train_index], Yi[test_index]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    accuracyRF = accuracyRandomForest(X_train, X_test, Y_train, Y_test)
    accuracyDT = accuracyDecisionTree(X_train, X_test, Y_train, Y_test)
    accuracyKN = accuracyKNN(X_train, X_test, Y_train, Y_test)
    accuracySVM = accuracySVC(X_train, X_test, Y_train, Y_test)
    accuracyNB = accuracyNaiveBayes(X_train, X_test, Y_train, Y_test)
    accuracyADA = accuracyAdaBoost(X_train, X_test, Y_train, Y_test)

    acc_values = [accuracyRF, accuracyDT, accuracyKN, accuracySVM, accuracyNB, accuracyADA]
    accuracy_finali.append(acc_values)  

print(accuracy_finali)

for train_index, test_index in shuffle_split.split(Wi, Yi):
        X_train, X_test = Wi[train_index], Wi[test_index]
        Y_train, Y_test = Yi[train_index], Yi[test_index]

#accED = accuracyED(X_train, X_test, Y_train, Y_test)
#print(accED)
#accDTW = accuracyDTW(X_train, X_test, Y_train, Y_test)
#print(accDTW)






# GRAFICO

pos = np.arange(len(accuracy_finali))
width = 0.12

plt.barh(pos + width*2, [y[0] for y in accuracy_finali], width, label='Random Forest', color='mediumseagreen')
plt.barh(pos + width, [y[1] for y in accuracy_finali], width, label='C4.5', color='yellow')
plt.barh(pos, [y[2] for y in accuracy_finali], width, label='KNN', color='orangered')
plt.barh(pos - width, [y[3] for y in accuracy_finali], width, label='SVM', color='royalblue')
plt.barh(pos - width*2, [y[4] for y in accuracy_finali], width, label='Naive Bayes', color='orchid')
plt.barh(pos - width*3, [y[5] for y in accuracy_finali], width, label='Logistic Regression', color='orange')

plt.yticks(pos, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'))
plt.xlabel('ACCURACY')
plt.xlim([0,1])
plt.ylabel('# FEATURES')
plt.title('URBAN OBSERVATORY')
plt.legend(loc='best')

plt.show()


