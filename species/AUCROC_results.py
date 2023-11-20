import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from analysis.txt

# overall, smallest distn, largest dist, densest, sparsest
FFNN_ROCAUC = [0.8895894828146357, 0.9829554440674312,  0.7230332612557422, 0.9510967601941662, 0.9150961871525265]
FFNN_PRAUC = [0.10934626640298879, 0.0002201976570877064, 0.30585417502055734, 0.10632347890778256, 0.008712417338564399]
FFNN_fscore = [0.25614824215132737, 0.0028110863464258238, 0.4540478790434726, 0.2721734087043962,  0.0245708445313523]
FFNN_cohenkappa = [0.950000051870154, 0.8830587631300217, 0.950001546980525, 0.9500334065532293, 0.950110233841554]

KNN_ROCAUC = [0.8105266236665908, 0.9923438361031535, 0.7835499803073702, 0.7710157704264592, 0.818944868850393]
KNN_PRAUC = [0.10453267822424134, 0.0022035294497383183, 0.29782065934594326, 0.14434688894704978, 0.011005006324878405]
KNN_fscore = [0.33481772888884875, 0.22375733828938965, 0.3523393918527871, 0.3962215559352821, 0.10830464523587369]
KNN_cohenkappa = [0.9748110250065037, 0.9975406922969142, 0.7709815489499279, 0.9680356234185584, 0.9922713908402716]

RF_ROCAUC = [0.921315129, 0.993952086, 0.789225301, 0.91294279, 0.899878905]
RF_PRAUC = [0.179543834, 0.018095636, 0.259446706, 0.291197765, 0.037386966]
RF_fscore = [0.337344274, 0.07107438, 0.511736309, 0.299561935, 0.0415294]
RF_cohenkappa = [0.979306, 0.99916, 0.731144, 0.977855, 0.993955]

L_ROCAUC = [0.862816068154488, 0.8763652374371139, 0.7032951497141235, 0.8589511432088969, 0.8951567952497239]
L_PRAUC = [0.012528571301748968, 0.0, 0.17104219931619918, 0.006631496842779755, 0.0004583145871513983]
L_fscore = [0.1026727110688724,  0.0006248172600042169, 0.33579763108879074, 0.18635430025443672, 0.011144652755728273]
L_cohenkappa = [0.8795202078873043, 0.8763100036702353, 0.7157382696718896, 0.8376943659152616, 0.8949911927644332]

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

df_F = pd.DataFrame({
    'FF-Neural Network': FFNN_fscore,
    'K-Nearest Neighbours': KNN_fscore,
    'Random Forest': RF_fscore,
    'Logistic Regression': L_fscore
}, index=FFNN_distns)

df_ck = pd.DataFrame({
    'FF-Neural Network': FFNN_cohenkappa,
    'K-Nearest Neighbours': KNN_cohenkappa,
    'Random Forest': RF_cohenkappa,
    'Logistic Regression': L_cohenkappa
}, index=FFNN_distns)

short_label = [name[:2] + '.' for name in df_PR.index]

custom_palette = sns.color_palette("Dark2")  

sns.set_palette(custom_palette)



df_ROC.plot(kind='bar', rot=0, width = 0.7, figsize=(10,5), legend=False)
plt.xlabel('Species Distribution type')
plt.ylabel('AUC-ROC')
plt.ylim(0.6, 1.0) #Added this, allows for easier comparison but would have to be mentioned in the report


df_F.plot(kind='bar', rot=0, width = 0.7, figsize=(10,5), legend=False)
plt.xlabel('Species Distribution type')
plt.ylabel('F-Score')


df_ck.plot(kind='bar', rot=0, width = 0.7, figsize=(10,5), legend=False)
plt.xlabel('Species Distribution type')
plt.ylabel('Cohen Kappa')
plt.ylim(0.6, 1.0) #Added this, allows for easier comparison but would have to be mentioned in the report


df_PR.plot(kind='bar', rot=0, width=0.7, figsize=(10, 5))
plt.xlabel('Species Distribution type')
plt.ylabel('AUC-PR')
plt.legend(title='Classifiers', bbox_to_anchor = (1.15, 1.15))

plt.tight_layout()


plt.show()