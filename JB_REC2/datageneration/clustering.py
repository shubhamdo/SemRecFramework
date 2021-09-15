from itertools import combinations
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.random import uniform
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from JB_REC2.connections.mongoConnection import insertCollection, getCollection
from JB_REC2.connections.neoconnection import connectToNeo4J
from sklearn import metrics
from sklearn.decomposition import PCA


def eval_hdbscan(df):
    df_sample = df.sample(n=30000)
    df_1 = df[df.index.isin(df_sample.index)].copy()
    df_2 = df_1.copy()
    csrMat = vectorizeData(df_sample['header.jobTitle'])
    # csrMat = vectorizeData(df_sample['job.description'])
    # [x] Cluster

    calinskis = []
    silhouettes = []
    min_cluster_size = []
    davies_bouldin = []
    for i in range(0, 100, 10):
        hdbsca, clusterer = get_hdbscan(df_1, csrMat, min_cluster_size=i + 2)

        cal = metrics.calinski_harabasz_score(csrMat.toarray(), df_1['cluster'])
        silhouet = metrics.silhouette_score(csrMat.toarray(), df_1['cluster'])
        dav = metrics.davies_bouldin_score(csrMat.toarray(), df_1['cluster'])

        calinskis.append(cal)
        silhouettes.append(silhouet)
        davies_bouldin.append(dav)

        # calinskis.append(metrics.calinski_harabasz_score(data2, data['cluster']))
        # silhouettes.append(metrics.silhouette_score(data2, data['cluster']))
        min_cluster_size.append(i + 2)
        print(cal, silhouet, i + 2, dav)
    return df_1, csrMat


def hdbscan_visualize(df_1, csrMat, min_cluster):
    hdbsca, clusterer = get_hdbscan(df_1, csrMat, min_cluster_size=min_cluster)
    pca = PCA(n_components=2).fit(csrMat.toarray())
    datapoint = pca.transform(csrMat.toarray())
    plt.figure

    color_palette = sns.color_palette('deep', 1000)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], s=50, linewidth=0, alpha=0.25, c=cluster_member_colors)
    plot_clustering(datapoint)


def eval_kmeans(df):
    df_sample = df.sample(n=30000)
    df_1 = df[df.index.isin(df_sample.index)].copy()
    df_2 = df_1.copy()
    csrMat = vectorizeData(df_sample['header.jobTitle'])
    # csrMat = vectorizeData(df_sample['job.description'])
    # [x] Cluster
    elbow = []
    calinskis = []
    silhouettes = []
    number_clusters = []
    i = 1
    for i in range(10):
        temp1, temp2 = get_kmeans(df_1, csrMat, i + 2)
        # kmeans.append(data.merge(temp1['cluster']))
        elbow.append(temp2)
        cal = metrics.calinski_harabasz_score(csrMat.toarray(), df_1['cluster'])
        silhouet = metrics.silhouette_score(csrMat.toarray(), df_1['cluster'])

        calinskis.append(cal)
        silhouettes.append(silhouet)
        number_clusters.append(i + 2)
        print(cal, silhouet)

    plt.plot(elbow, 'ro-', label="Elbow")
    plt.title("KMeans Elbow")
    plt.show()

    plt.plot(number_clusters, calinskis, 'ro-', label="KMeans Ralinski Harabasz Score")
    plt.title("KMeans Calinski Harabasz Score")
    plt.xlabel("number of clusters")
    plt.show()

    plt.plot(number_clusters, silhouettes, 'ro-', label="KMeans Silhouette Score")
    plt.title("KMeans Silhouette Score")
    plt.xlabel("number of clusters")
    plt.show()

    plot_clustering(df_1)
    return df_1, csrMat

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


def get_kmeans(data, csr, n_clusters=3):
    """ Do kmeans clustering and return clustered data """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300)
    # vals = data.iloc[:, 0:].values
    y_pred = kmeans.fit_predict(csr)
    data["cluster"] = y_pred
    return data, kmeans.inertia_


def get_hdbscan(data, csrMat, min_cluster_size=4):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    y_pred = clusterer.fit_predict(csrMat)
    data["cluster"] = y_pred.tolist()

    # hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    # vals = data.iloc[:, 0:].values
    # y_pred = hdb.fit_predict(StandardScaler().fit_transform(vals))
    # data["cluster"] = y_pred
    return data, clusterer


def vectorizeData(text):
    vec = CountVectorizer()
    X = vec.fit_transform(text)
    return X


def concatenateStrings(df):
    # title, description = df['header.jobTitle'], df['job.description']
    # cnString = title + " " + description
    df['test'] = ""
    for index, row in df.iterrows():
        title = row[0]
        description = row[1]
        cnString = title + " " + description
        df.at[index, 'test'] = cnString

    return df


def DataPairGeneratorHDBSCAN(df, clusterSize):
    dataSize = df.shape[0]
    while dataSize >= 0:
        nor = 10000
        if dataSize <= 10000:
            nor = dataSize
        # [x] Get Random 10000 Records
        df_sample = df.sample(n=nor)
        # [x] Filter Records in original records
        df = df[~df.index.isin(df_sample.index)]
        dataSize = df.shape[0]
        if dataSize == 0:
            break
        # [x] Convert into vectors
        csrMat = vectorizeData(df_sample['header.jobTitle'])

        # [x] Cluster
        clusterer = hdbscan.HDBSCAN(min_cluster_size=clusterSize)
        clusterer.fit_predict(csrMat)
        df_sample['cluster'] = clusterer.labels_.tolist()
        df_sample = df_sample[df_sample['cluster'] != -1]

        # [x] Create Positive Samples and Negative Samples
        clusters = set(df_sample['cluster'].tolist())
        for cluster in clusters:
            currentCluster = df_sample[df_sample['cluster'] == cluster]

            if currentCluster.shape[0] > 100:
                currentCluster = currentCluster.sample(n=100)

            otherCluster = df_sample[df_sample['cluster'] != cluster]  # Required for Negative Records
            # otherCluster = df_sample[~df_sample.index.isin(currentCluster.index)]  # Required for Negative Records
            lenOfCurrentCluster = currentCluster.shape[0]
            lenOfOtherCluster = otherCluster.shape[0]
            if lenOfCurrentCluster > lenOfOtherCluster:
                lenOfCurrentCluster = lenOfOtherCluster
            print("The current cluster is: {} and length of cluster is: {} ".format(cluster, lenOfCurrentCluster))
            otherCluster = otherCluster.sample(n=lenOfCurrentCluster)  # Gets num of records as the positive

            currentJobDescriptions = currentCluster['job.description']
            otherJobDescriptions = otherCluster['job.description']

            for combination in combinations(otherJobDescriptions, 2):
                # print(combination)
                record = dict()
                record['inputA'] = combination[0].strip()
                record['inputB'] = combination[1].strip()
                record['similarity'] = uniform(0.1, 0.4)
                negativeRecords = [record]
                negativeRecords = pd.DataFrame(negativeRecords)
                negativeRecords = negativeRecords.drop_duplicates()
                insertCollection("trainingData", "negativeSamples", negativeRecords)

            for combination in combinations(currentJobDescriptions, 2):
                record = dict()
                record['inputA'] = combination[0].strip()
                record['inputB'] = combination[1].strip()
                record['similarity'] = uniform(0.6, 1.0)
                positiveRecords = [record]
                positiveRecords = pd.DataFrame(positiveRecords)
                positiveRecords = positiveRecords.drop_duplicates()
                insertCollection("trainingData", "positiveSamples", positiveRecords)


def DataPairGeneratorKmeans(df, noOfK):
    dataSize = df.shape[0]
    while dataSize >= 0:
        nor = 10000
        if dataSize <= 10000:
            nor = dataSize
        # [x] Get Random 10000 Records
        df_sample = df.sample(n=nor)
        # [x] Filter Records in original records
        df = df[~df.index.isin(df_sample.index)]
        dataSize = df.shape[0]
        if dataSize == 0:
            break
        # [x] Convert into vectors
        csrMat = vectorizeData(df_sample['header.jobTitle'])

        # [x] Cluster
        clusterer = KMeans(n_clusters=noOfK, random_state=0)
        clusterer.fit_predict(csrMat)
        df_sample['cluster'] = clusterer.labels_.tolist()
        df_sample = df_sample[df_sample['cluster'] != -1]

        # [x] Create Positive Samples and Negative Samples
        clusters = set(df_sample['cluster'].tolist())
        for cluster in clusters:
            currentCluster = df_sample[df_sample['cluster'] == cluster]

            if currentCluster.shape[0] > 100:
                currentCluster = currentCluster.sample(n=100)

            otherCluster = df_sample[df_sample['cluster'] != cluster]  # Required for Negative Records
            # otherCluster = df_sample[~df_sample.index.isin(currentCluster.index)]  # Required for Negative Records
            lenOfCurrentCluster = currentCluster.shape[0]
            lenOfOtherCluster = otherCluster.shape[0]
            if lenOfCurrentCluster > lenOfOtherCluster:
                lenOfCurrentCluster = lenOfOtherCluster
            print("The current cluster is: {} and length of cluster is: {} ".format(cluster, lenOfCurrentCluster))
            otherCluster = otherCluster.sample(n=lenOfCurrentCluster)  # Gets num of records as the positive

            currentJobDescriptions = currentCluster['job.description']
            otherJobDescriptions = otherCluster['job.description']

            for combination in combinations(otherJobDescriptions, 2):
                # print(combination)
                record = dict()
                record['inputA'] = combination[0].strip()
                record['inputB'] = combination[1].strip()
                record['similarity'] = uniform(0.1, 0.4)
                negativeRecords = [record]
                negativeRecords = pd.DataFrame(negativeRecords)
                negativeRecords = negativeRecords.drop_duplicates()
                insertCollection("trainingData", "negativeSamples", negativeRecords)

            for combination in combinations(currentJobDescriptions, 2):
                record = dict()
                record['inputA'] = combination[0].strip()
                record['inputB'] = combination[1].strip()
                record['similarity'] = uniform(0.6, 1.0)
                positiveRecords = [record]
                positiveRecords = pd.DataFrame(positiveRecords)
                positiveRecords = positiveRecords.drop_duplicates()
                insertCollection("trainingData", "positiveSamples", positiveRecords)


if __name__ == "__main__":
    # Connect to DB
    graph = connectToNeo4J()
    graph.nodes.match("JD").count()

    # Retrieve Data from Neo4J
    jdNodes = graph.nodes.match("JD").all()

    # Convert to Dataframe
    # df = pd.read_csv("H:/Thesis/Output_Data/cluster_all.csv", sep=";")
    df = getCollection("thesisDb_Crawlers", "glassdoorListingsClean")
    df = df[['job.listingId.long', 'header.jobTitle', 'job.description']]
    df = df.set_index('job.listingId.long')

    # Depending on the eval code we decide to create pairs, once evals are cleared, parameters are updated for final
    # Pairs of Data
    # Evaluation of Clustering Method - HDBSCAN
    df_1, csrMat = eval_hdbscan(df)
    hdbscan_visualize(df_1, csrMat, min_cluster=12)

    # Evaluation of Clustering Method - KMeans & Visualizations
    eval_kmeans(df)

    # Data Pair Generator using HDBSCAN
    DataPairGeneratorHDBSCAN(df, 200)

    # Data Pair Generator using Kmeans
    DataPairGeneratorKmeans(df, 10)
