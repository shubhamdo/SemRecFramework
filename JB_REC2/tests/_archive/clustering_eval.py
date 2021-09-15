import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.datasets import make_blobs


def generate_points():
    """ Generate random points. """
    centers = []
    centers.append([[0.75, -0.75]])
    centers.append([[-0.75, -0.75]])
    centers.append([[0.75, 0.75]])
    # print(centers)
    datas = [None] * len(centers)
    labels_true = [None] * len(centers)
    datas[0], labels_true[0] = make_blobs(n_samples=40, centers=centers[0], cluster_std=0.4)
    datas[1], labels_true[1] = make_blobs(n_samples=15, centers=centers[1], cluster_std=0.10)
    datas[2], labels_true[2] = make_blobs(n_samples=25, centers=centers[2], cluster_std=0.20)
    print(datas[0])
    print(len(datas))
    for i in range(len(datas) - 1):
        data = np.append(datas[0], datas[i + 1], axis=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    # print(data)
    # filee = open("C:/Users/shubham/PycharmProjects/JB_REC/JB_REC/tests/points.txt", "w", newline = '')
    data.to_csv("C:/Users/shubham/PycharmProjects/JB_REC2/JB_REC2/tests/points.txt", sep=',', index=False)


def plot_clustering(data, k=None, vars=None):
    """ Plot the clustered data. """
    if vars == None:
        cols = list(data.columns)
    else:
        vars.append('cluster')
        vars = set(vars)
        vars = list(vars)
        cols = vars
    g = sns.pairplot(data[cols], hue='cluster', diag_kind='hist')
    if not k == None:
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(k + 2)
        g.fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05)
    plt.show()


def get_kmeans(data, n_clusters=3):
    """ Do kmeans clustering and return clustered data """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300)
    vals = data.iloc[:, 0:].values
    y_pred = kmeans.fit_predict(StandardScaler().fit_transform(vals))
    data["cluster"] = y_pred
    return data, kmeans.inertia_


def get_hdbscan(data, min_cluster_size=4):
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    vals = data.iloc[:, 0:].values
    y_pred = hdb.fit_predict(StandardScaler().fit_transform(vals))
    data["cluster"] = y_pred
    return data, hdb


data = pd.read_csv("C:/Users/shubham/PycharmProjects/JB_REC/JB_REC/tests/points.txt")
data2 = data
# plt.plot(data['y'], data['x'], 'o')
# plt.title("Data")
# plt.show()

kmeans = []
elbow = []
calinskis = []
silhouettes = []
number_clusters = []
i = 1
for i in range(10):
    temp1, temp2 = get_kmeans(data, i + 2)
    kmeans.append(data.merge(temp1['cluster']))
    elbow.append(temp2)
    print(metrics.calinski_harabasz_score(data2, data['cluster']),
          metrics.silhouette_score(data2, data['cluster']), (i + 2))
    calinskis.append(metrics.calinski_harabasz_score(data2, data['cluster']))
    silhouettes.append(metrics.silhouette_score(data2, data['cluster']))
    number_clusters.append(i + 2)
plt.plot(elbow, 'ro-', label="Elbow")
plt.title("KMeans Elbow")
plt.show()
kmean, temp = get_kmeans(data)
kmean = data.merge(kmean['cluster'])

plt.plot(number_clusters, calinskis, 'ro-', label="KMeans Ralinski Harabasz Score")
plt.title("KMeans Calinski Harabasz Score")
plt.xlabel("number of clusters")
plt.show()

plt.plot(number_clusters, silhouettes, 'ro-', label="KMeans Silhouette Score")
plt.title("KMeans Silhouette Score")
plt.xlabel("number of clusters")
plt.show()

plot_clustering(kmean)

#####################################################################################

calinskis = []
silhouettes = []
min_cluster_size = []
for i in range(8):
    hdbsca, clusterer = get_hdbscan(data, i + 3)
    hdbsca = data.merge(hdbsca)
    calinskis.append(metrics.calinski_harabasz_score(data2, data['cluster']))
    silhouettes.append(metrics.silhouette_score(data2, data['cluster']))
    min_cluster_size.append(i + 3)
    print(metrics.calinski_harabasz_score(data2, data['cluster']), \
          metrics.silhouette_score(data2, data['cluster']))
hdbsca, clusterer = get_hdbscan(data)
hdbsca = data.merge(hdbsca['cluster'])
plt.plot(min_cluster_size, calinskis, 'ro-', label="HDBSCAN Ralinski Harabasz Score")
plt.title("HDBSCAN Calinski Harabasz Score")
plt.xlabel("minimum cluster size")
plt.show()

plt.plot(min_cluster_size, silhouettes, 'ro-', label="HDBSCAN Silhouette Score")
plt.title("HDBSCAN Silhouette Score")
plt.xlabel("minimum cluster size")
plt.show()

plot_clustering(hdbsca)
