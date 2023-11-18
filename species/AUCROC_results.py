import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from analysis.txt

# overall, smallest distn, largest dist, densest, sparsest
FFNN_ROCAUC = [0, 0.9973299627737877, 0.7398151304130265, 0.9080670271172481, 0.8838673788189455]
FFNN_PRAUC = [0, 0.0009417050896884918, 0.3402472246427667, 0.3566891002006113, 0.02311410729356262]
FFNN_fscore = []
FFNN_cohenkappa = []

KNN_ROCAUC = [0, 0.8822738011960112, 0.5934899480544569, 0.5413054824045366, 0.6235921885184947]
KNN_PRAUC = [0, 0.09324385729672427, 0.2777463874596238, 0.35763915230197807, 0.1267579860034622]
KNN_fscore = []
KNN_cohenkappa = []

RF_ROCAUC = [0.91673283, 0.9988412416134949, 0.7855063269571104, 0.9069871316479985, 0.8877357900895622]
RF_PRAUC = [0, 0.0009127801966411176, 0.3105575887116155, 0.32244796706285755, 0.035381587451708024]
RF_fscore = []
RF_cohenkappa = []

L_ROCAUC = [0, 0.8710394655199822, 0.650671390779345, 0.8035460350985414, 0.8134056042496407]
L_PRAUC = [0, 8.130388823287408e-05, 0.12755397242139097, 0.0022851718828855853, 0.0]
L_fscore = []
L_cohenkappa = []

FFNN_distns = ['Overall' ,'Smallest', 'Largest', 'Densest', 'Sparsest']
sns.set_theme()

df_ROC = pd.DataFrame({
    'FF-Neural Network': FFNN_ROCAUC,
    'K-Nearest Neighbours': KNN_ROCAUC,
    'Random Forest': RF_ROCAUC,
    'Logistic Regression': L_ROCAUC
}, index=FFNN_distns)

df_PR = pd.DataFrame({
    'FF-Neural Network': FFNN_PRAUC,
    'K-Nearest Neighbours': KNN_PRAUC,
    'Random Forest': RF_PRAUC,
    'Logistic Regression': L_PRAUC
}, index=FFNN_distns)

short_label = [name[:2] + '.' for name in df_PR.index]

custom_palette = sns.color_palette("gray")  

# sns.set_palette(custom_palette)



df_ROC.plot(kind='bar', rot=0, width = 0.7, figsize=(10,5), legend=False)
plt.xlabel('Species Distribution type')
plt.ylabel('AUC-ROC')


df_PR.plot(kind='bar', rot=0, width=0.7, figsize=(10, 5))
plt.xlabel('Species Distribution type')
plt.ylabel('AUC-PR')
plt.legend(title='Classifiers', bbox_to_anchor = (1.15, 1.15))


plt.show()