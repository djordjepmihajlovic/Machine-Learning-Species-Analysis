import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_train = np.load('species_train.npz', mmap_mode="r")
species_names = dict(zip(data_train['taxon_ids'], data_train['taxon_names']))  # latin names of species 

FFNN_AUC = [0.921257146594979, 0.8962831204206212, 0.9937456615299181, 0.9240093776825322, 0.8434306888124863]
KNN_AUC = [0.9, 0.8, 0.99, 0.7, 0.9]
DT_AUC = [0.8, 0.9, 0.7, 0.9, 0.99]
FFNN_species = [12716, 4535, 35990, 13851, 43567]
sns.set_theme()

df = pd.DataFrame({
    'FFNN': FFNN_AUC,
    'KNN': KNN_AUC,
    'DT': DT_AUC
}, index=[species_names[i] for i in FFNN_species])

short_label = [name[:2] + '.' for name in df.index]

custom_palette = sns.color_palette("muted")  

sns.set_palette(custom_palette)
df.plot(kind='bar')
plt.xlabel('Species')
plt.xticks(range(len(df.index)), short_label, rotation=0, ha='right')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC Values for Different Classifiers vs. Different Species')
plt.legend(title='Classifiers', loc='lower right')
plt.show()