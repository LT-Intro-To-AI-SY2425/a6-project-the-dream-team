import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data =pd.read_csv("all_players.csv")
df=data[data["POS"]=="F"]


data=df[["G","A","SHTS","SOG","GWG","GWA"]]
scale=StandardScaler().fit_transform(data)
c=4
km=KMeans(n_clusters=c).fit(scale)
cen=km.cluster_centers_
lab=km.labels_
plt.figure(figsize=(5,4))
for i in range(c):
    clus=scale[lab==i]
    plt.scatter(clus[:,0],clus[:,1])
plt.scatter(cen[:,0],cen[:,1],marker="*",s=200,c="black",label="centroid")
plt.xlabel("forward Stats")
plt.ylabel("forward Prediction")
plt.show()

