from sklearn.decomposition import PCA
import pandas as pd


data = pd.read_csv('change_1.csv')


pca = PCA(n_components=3)
pca.fit(data)
print(pca.explained_variance_ratio_)

pca.transform(data)

