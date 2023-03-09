import pandas as pd
import matplotlib.pyplot as plt

#PCA
df = pd.read_csv('winequalityN.csv')
#separating out the color column
df = df.iloc[:, 1:]

#turning quality values into booleans
df['quality'] = df['quality'].apply(lambda x: 1 if x > 7 else 0)

#creating data and labels
df_data = df.iloc[:, :-1]
df_label = df.iloc[:, -1]

#implementing PCA on df
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#normalizing data
sc= StandardScaler()
df_data = sc.fit_transform(df_data)

pca = PCA(n_components=2)
df_data = pca.fit_transform(df_data)

fig, ax = plt.subplots(figsize=(10,10))
colors = ['pink', 'red']
legend = ['Not Quality', 'Quality']

for i in range(len(legend)):
    ax.scatter(df_data[df_label == i, 0], df_data[df_label == i, 1], c=colors[i], s=60)

ax.set_xlabel('Principal Component - 1', fontsize=15)
ax.set_ylabel('Principal Component - 2', fontsize=15)
ax.set_title('PCA of Wine Quality Dataset', fontsize=20)
ax.legend(legend, fontsize=15)
plt.show()




#TSNE
df = pd.read_csv('winequalityN.csv')
#separating out the color column
df = df.iloc[:, 1:]

#turning quality values into booleans
df['quality'] = df['quality'].apply(lambda x: 1 if x > 7 else 0)

#creating data and labels
df_data = df.iloc[:, :-1]
df_label = df.iloc[:, -1]

from sklearn.manifold import TSNE

#normalizing data
sc= StandardScaler()
df_data = sc.fit_transform(df_data)

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
fig, ax = plt.subplots(figsize=(10,10))
colors = ['pink', 'red']
legend = ['Not Quality', 'Quality']

for i in range(len(legend)):
    ax.scatter(df_data[df_label == i, 0], df_data[df_label == i, 1], c=colors[i], s=60)

ax.set_xlabel('Principal Component - 1', fontsize=15)
ax.set_ylabel('Principal Component - 2', fontsize=15)
ax.set_title('t-SNE of Wine Quality Dataset', fontsize=20)
ax.legend(legend, fontsize=15)
plt.show()





#PCA then graphing using the 8th and 9th components
df = pd.read_csv('winequalityN.csv')
#separating out the color column
df = df.iloc[:, 1:]
#turning quality values into booleans
df['quality'] = df['quality'].apply(lambda x: 1 if x > 7 else 0)

#creating data and labels
df_data = df.iloc[:, :-1]
df_label = df.iloc[:, -1]

#implementing PCA on df
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#normalizing data
sc= StandardScaler()
df_data = sc.fit_transform(df_data)

pca = PCA(n_components=11)
df_data = pca.fit_transform(df_data)

#choosing the 8th and 9th PCs
df_data_89 = df_data[:, 7:9]

fig, ax = plt.subplots(figsize=(10,10))
colors = ['pink', 'red']
legend = ['Not Quality', 'Quality']

for i in range(len(legend)):
     ax.scatter(df_data_89[df_label == i, 0], df_data_89[df_label == i, 1], c=colors[i], s=60)

ax.set_xlabel('Principal Component - 8', fontsize=15)
ax.set_ylabel('Principal Component - 9', fontsize=15)
ax.set_title('PCA of Wine Quality Dataset', fontsize=20)
ax.legend(legend, fontsize=15)
plt.show()