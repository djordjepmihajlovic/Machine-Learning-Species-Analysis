import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from analysis.txt

# overall, smallest distn, largest dist, densest, sparsest
FFNN_ROCAUC = [0, 0.9973299627737877, 0.7398151304130265, 0.9080670271172481, 0.8838673788189455]
FFNN_PRAUC = [0, 0.0009417050896884918, 0.3402472246427667, 0.3566891002006113, 0.02311410729356262]
KNN_ROCAUC = [0, 0.8822738011960112, 0.5934899480544569, 0.5413054824045366, 0.6235921885184947]
KNN_PRAUC = []
RF_ROCAUC = [0.91673283, 0.9988412416134949, 0.7855063269571104, 0.9069871316479985, 0.8877357900895622]
DT_PRAUC = []
L_ROCAUC = [0, 0.8710394655199822, 0.650671390779345, 0.8035460350985414, 0.8134056042496407]
L_PRAUC = []

FFNN_distns = ['Overall' ,'Smallest', 'Largest', 'Densest', 'Sparsest']
sns.set_theme()

df = pd.DataFrame({
    'FFNN_ROCAUC': FFNN_ROCAUC,
    'KNN_ROCAUC': KNN_ROCAUC,
    'RF_ROCAUC': RF_ROCAUC,
    'L_ROCAUC': L_ROCAUC
}, index=FFNN_distns)

short_label = [name[:2] + '.' for name in df.index]

custom_palette = sns.color_palette("gray")  

# sns.set_palette(custom_palette)
df.plot(kind='bar', rot=0)

plt.xlabel('Species')
# plt.xticks(range(len(df.index)), short_label, rotation=0, ha='right')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC Values for Different Classifiers vs. Different Species')
plt.legend(title='Classifiers', loc='lower right')
plt.show()