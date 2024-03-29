import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from analysis.txt

# overall, smallest distn, largest dist, densest, sparsest
FFNN_ROCAUC = [0.8895894828146357, 0.9829554440674312,  0.7230332612557422, 0.9510967601941662, 0.9150961871525265]
FFNN_PRAUC = [0.10934626640298879, 0.0002201976570877064, 0.30585417502055734, 0.10632347890778256, 0.008712417338564399]
tot_FFNN_f2score = [0.2663643, 0.0060272307646420265, 0.5444788633629031, 0.43453053664890745,  0.0804239993134672]
FFNN_cohenkappa = [0.950000051870154, 0.8830587631300217, 0.950001546980525, 0.9500334065532293, 0.950110233841554]

KNN_ROCAUC = [0.8105266236665908, 0.9923438361031535, 0.7835499803073702, 0.7710157704264592, 0.818944868850393]
KNN_PRAUC = [0.10453267822424134, 0.0022035294497383183, 0.29782065934594326, 0.14434688894704978, 0.011005006324878405]
tot_KNN_f2score = [0.38185409595302783,  0.32845834380035754,  0.3189961844741406, 0.40506809862754345, 0.1943711761687747]
KNN_cohenkappa = [0.9748110250065037, 0.9975406922969142, 0.7709815489499279, 0.9680356234185584, 0.9922713908402716]

RF_ROCAUC = [0.921315129, 0.993952086, 0.789225301, 0.91294279, 0.899878905]
RF_PRAUC = [0.179543834, 0.018095636, 0.259446706, 0.291197765, 0.037386966]
tot_RF_f2score = [0.367623, 0.5894656416106102, 0.3984667816135756, 0.24653704580032917, 0.23601243058401625]
RF_cohenkappa = [0.979306, 0.99916, 0.731144, 0.977855, 0.993955]

L_ROCAUC = [0.862816068154488, 0.8763652374371139, 0.7032951497141235, 0.8589511432088969, 0.8951567952497239]
L_PRAUC = [0.012528571301748968, 0.0, 0.17104219931619918, 0.006631496842779755, 0.0004583145871513983]
tot_L_f2score = [0.18278362619081184,  0.0015595448659486764, 0.39050844636939136, 0.3395025949323818, 0.02718416316064351]
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

FFNN_f2score = [0.0060272307646420265, 0.5444788633629031, 0.43453053664890745,  0.0804239993134672]
KNN_f2score = [0.2867374951897565, 0.024911769657153155, 0.35074926448794064, 0.17816085215378735]
LR_f2score = [0.0015595448659486764, 0.4526448860622701, 0.3395025949323818,  0.02718416316064351]
RF_f2score = [0.6164291331654295, 0.256452371, 0.24535449960177963, 0.23204482537866267]

# 8 feature trained data F2 scores

FFNN_8_f2score = [0.021896046161400416, 0.48574291976412265, 0.6387388800542697,  0.04616623553546854]
KNN_8_f2score = [0.26010530681267463, 0.024155189874221435, 0.38982156808148005,  0.22175545981134115]
LR_8_f2score = [0.0973767368522539, 0.17726366456166806, 0.33663020908651947,  0.09505249198503463]
RF_8_f2score = [0.4471554352737359, 0.24655023079042418, 0.25189590099648523, 0.214568242]





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
    'FF-Neural Network': tot_FFNN_f2score,
    'K-Nearest Neighbours': tot_KNN_f2score,
    'Random Forest': tot_RF_f2score,
    'Logistic Regression': tot_L_f2score
}, index=FFNN_distns)

df_ck = pd.DataFrame({
    'FF-Neural Network': FFNN_cohenkappa,
    'K-Nearest Neighbours': KNN_cohenkappa,
    'Random Forest': RF_cohenkappa,
    'Logistic Regression': L_cohenkappa
}, index=FFNN_distns)


df_compare = pd.DataFrame({
    'FFNN difference': np.array(FFNN_8_f2score)-np.array(FFNN_f2score),
    'KNN difference' : np.array(KNN_8_f2score) - np.array(KNN_f2score),
    'RF difference' : np.array(RF_8_f2score) - np.array(RF_f2score),
    'LR difference' : np.array(LR_8_f2score) - np.array(LR_f2score)

}, index=distns)

short_label = [name[:2] + '.' for name in df_PR.index]

custom_palette = sns.color_palette("Set1")  

sns.set_style("dark")
sns.set_palette(custom_palette)

df_compare.plot(kind = 'bar', rot= 0, width = 0.7, figsize=(10,5), linewidth=2.5, edgecolor = "black")
ax = plt.gca()
# Label the bars with the differences
ax.bar_label(ax.containers[0], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[1], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[2], fontsize=10, fmt='%.2f')
ax.bar_label(ax.containers[2], fontsize=10, fmt='%.2f')

ax.axhline(0, color='grey', linewidth=2.5, linestyle='-')
plt.ylabel('F2-Score Improvement')
plt.ylim(-0.4, 0.4)


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