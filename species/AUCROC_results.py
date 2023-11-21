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

# 8 feature trained data (land data only - ommitting 4636 and 4146 from largest distn)

FFNN_8_ROCAUC = [0.902579477951429, 0.8399361655091986, 0.9475623289230691, 0.7714732423744892]
FFNN_8_PRAUC = [0.0008611267993687207, 0.13176591437055427, 0.26639615443337517, 0.01771721266463815]
FFNN_8_fscore = [0.010320332935460402, 0.43772246993119124, 0.6205332591505883, 0.058485203311628695]
FFNN_8_cohenkappa = [0.9504894726118973, 0.9500066192512364, 0.9500019602783963 , 0.9500104658917431]

KNN_8_ROCAUC = [0.9025679301322349, 0.8078411448289917, 0.7977284833297411, 0.732629490793492]
KNN_8_PRAUC = [0.03218871014877396, 0.5887419983982521, 0.358678256099313, 0.1303710417984521]
KNN_8_fscore = [0.19717863821550816, 0.037701714386056566, 0.46514326296963027, 0.22651694049956786]
KNN_8_cohenkappa = [0.9993350384413091, 0.9330369349503131, 0.9781921283919512, 0.9989447214101542]

LR_8_ROCAUC = [0.9594794473595855, 0.7107350848436171, 0.9455188046204226, 0.8337121843394245]
LR_8_PRAUC = [0.003450760917457295, 0.46500235070484125, 0.5328091584127936, 0.07997415976781375]
LR_8_fscore = [0.04590529427188863, 0.24880538849485598, 0.3774708693463086, 0.10600674823370926]
LR_8_cohenkappa = [0.99824809411658, 0.924484831435183, 0.9800375917904873,0.9620638140420263]

# 2 feature trained data F2 scores (land data only - ommiting 4636 and 4146 from largest distn)

FFNN_f2score = [0.0028110863464258238, 0.4540478790434726, 0.2721734087043962,  0.0245708445313523]

KNN_f2score = [0.0028110863464258238, 0.4540478790434726, 0.2721734087043962,  0.0245708445313523]

LR_f2score = [0.0028110863464258238, 0.4540478790434726, 0.2721734087043962,  0.0245708445313523]

RF_f2score = [0.0028110863464258238, 0.4540478790434726, 0.2721734087043962,  0.0245708445313523]



FFNN_distns = ['Overall' ,'Smallest', 'Largest', 'Densest', 'Sparsest']
distns = ['Smallest', 'Largest', 'Densest', 'Sparsest']
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


df_compare = pd.DataFrame({
    'FFNN difference': np.array(FFNN_8_fscore)-np.array(FFNN_f2score),

    'KNN difference' : np.array(KNN_8_fscore) - np.array(KNN_f2score),

    'LR difference' : np.array(LR_8_fscore) - np.array(LR_f2score),

}, index=distns)

short_label = [name[:2] + '.' for name in df_PR.index]

custom_palette = sns.color_palette("Set1")  

sns.set_style("dark")
sns.set_palette(custom_palette)

df_compare.plot(kind = 'bar', rot= 0, linewidth=2.5, edgecolor = "black")
ax = plt.gca()
# Label the bars with the differences
ax.bar_label(ax.containers[0], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[1], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[2], fontsize=10, fmt='%.2f')


plt.tight_layout()

df_ROC.plot(kind='bar', rot=0, width = 0.7, figsize=(10,5), linewidth=2.5, edgecolor = "black")
# plt.xlabel('Species Distribution type')
# plt.ylabel('AUC-ROC')
plt.ylim(0.6, 1.05) #Added this, allows for easier comparison but would have to be mentioned in the report
ax = plt.gca()
ax.bar_label(ax.containers[0], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[1], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[2], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[3], fontsize=10, fmt='%.2f')
plt.tight_layout()

df_F.plot(kind='bar', rot=0, width = 0.7, figsize=(10,5), linewidth=2.5, edgecolor = "black")
# plt.xlabel('Species Distribution type')
plt.ylim(0, 0.6)

ax = plt.gca()
ax.bar_label(ax.containers[0], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[1], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[2], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[3], fontsize=10, fmt='%.2f')

# ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()


df_ck.plot(kind='bar', rot=0, width = 0.7, figsize=(10,5), linewidth=2.5, edgecolor = "black")
# plt.xlabel('Species Distribution type')
# plt.ylabel('Cohen Kappa')
plt.ylim(0.6, 1.05) #Added this, allows for easier comparison but would have to be mentioned in the report
ax = plt.gca()
ax.bar_label(ax.containers[0], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[1], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[2], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[3], fontsize=10, fmt='%.2f')

# ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.legend(loc = 'upper right')

df_PR.plot(kind='bar', rot=0, width=0.7, figsize=(10, 5), linewidth=2.5, edgecolor = "black")
# plt.xlabel('Species Distribution type')
plt.ylim(0, 0.6)

ax = plt.gca()
ax.bar_label(ax.containers[0], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[1], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[2], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[3], fontsize=10, fmt='%.2f')

# ax.set_xticks([])
# ax.set_yticks([])
plt.tight_layout()
# plt.legend(title='Classifiers', bbox_to_anchor = (1.15, 1.15))



plt.show()