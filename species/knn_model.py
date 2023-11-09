import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('cm_turdus.png')
    plt.show()

# necessary to get rid of annoying scipy warning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# load training data    
data = np.load('species/species_train.npz')
#data = np.load('species_train.npz')
train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

# test data
data_test = np.load('species/species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

# k nearest neighbours classifier
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(train_locs, train_ids)

"""
# coords of edinburgh city center
la = 55.953332
lo = -3.189101
print(species_names[knn.predict([[la,lo]])[0]])
"""
id_test = 12716
pos_inds = test_pos_inds[id_test]
true = np.zeros(len(test_locs))
true[pos_inds] = 1 

pred = knn.predict(test_locs)
tp = 0
fp = 0
tn = 0
fn = 0
for i in range(len(pred)):
    if pred[i] == id_test:
        pred[i] = 1
        if (pred[i] == true[i]):
            tp += 1
        else:
            fp += 1
    else:
        pred[i] = 0
        if (pred[i] == true[i]):
            tn += 1
        else:
            fn += 1

cm = confusion_matrix(true, pred)
print(cm)
print(tp)
print(fn)
print(fp)
print(tn)