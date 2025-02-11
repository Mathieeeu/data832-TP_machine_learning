import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns

import os

filename = "data/Country-data.csv"

def load_data(file_name: str) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    return data

def mean_and_variance_for_each_feature(data: pd.DataFrame) -> dict:
    res = {}
    for col in data.columns[1:]:
        res[col] = (data[col].mean(), data[col].var())
    return res

def correlation_matrix(data: pd.DataFrame) -> None:
    numeric_data = data.select_dtypes(include=[np.number]) 
    data_corr=numeric_data.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_corr,annot=True, cmap="magma")
    plt.title("Correlation Matrix")
    plt.show()

def plot_data(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    sns.histplot(data['child_mort'], ax=axes[0, 0])
    sns.histplot(data['exports'], ax=axes[0, 1])
    sns.histplot(data['health'], ax=axes[0, 2])
    sns.histplot(data['imports'], ax=axes[1, 0])
    sns.histplot(data['income'], ax=axes[1, 1])
    sns.histplot(data['inflation'], ax=axes[1, 2])
    sns.histplot(data['life_expec'], ax=axes[2, 0])
    sns.histplot(data['total_fer'], ax=axes[2, 1])
    sns.histplot(data['gdpp'], ax=axes[2, 2])

    plt.tight_layout()
    plt.show()

def plot_feature(data: pd.DataFrame, feature: str, nb_countries: int = 10, ascending: bool = False) -> None: 
    data_sorted = data.sort_values(by=[feature], ascending=ascending)
    data_sorted = data_sorted.head(nb_countries)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=data_sorted[feature], y=data_sorted['country'])
    plt.title(f"Top {nb_countries} countries with the highest {feature}" if not ascending else f"Top {nb_countries} countries with the lowest {feature}")
    plt.show()

def pipeline(data: pd.DataFrame) -> dict:
    """"
    Preprocessing : standardscaler
    Decomposition : PCA
    Clustering : KMeans (3 clusters : help needed, no help needed, may need help)
    """
    data_clean = data.drop('country', axis=1)
    pipe = sk.pipeline.Pipeline([('scaler',sk.preprocessing.StandardScaler()),
                                 ('pca', sk.decomposition.PCA(n_components=2)),
                                 ('kmeans', sk.cluster.KMeans(n_clusters=3))])
    pipe.fit(data_clean)

    # Predict the cluster for each data point
    res = pd.DataFrame({'country': data['country'], 'cluster': pipe.predict(data_clean)})
    return res

def plot_cluster_label_on_a_map_of_the_Earth(data: pd.DataFrame) -> None:
    if not os.path.isfile("data/countries.geojson"):
        world = gpd.read_file("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson")
        world.to_file("data/countries.geojson")
    else:
        world = gpd.read_file("data/countries.geojson")

    world = world.merge(data, how='left', left_on='ADMIN', right_on='country')
    world['cluster'] = world['cluster'].fillna(-1)
    colors = {0: '#beffc1', 1: '#aad0ff', 2: '#ff9a95', -1: '#aaaaaa'}
    world['color'] = world['cluster'].map(colors)
    world.plot(color=world['color'], legend=True, figsize=(15, 10))
    plt.title('Clusters of countries')
    plt.show()

if __name__ == '__main__':
    data = load_data(filename)
    features = mean_and_variance_for_each_feature(data)
    #correlation_matrix(data)

    #plot_data(data)
    # plot_feature(data, feature='gdpp', nb_countries=10, ascending=False)
    # plot_feature(data, feature='child_mort', nb_countries=10, ascending=True)
    # plot_feature(data, feature='income', nb_countries=10, ascending=False)

    clustered_data = pipeline(data)

    plot_cluster_label_on_a_map_of_the_Earth(clustered_data)

    for row in clustered_data.iterrows():
        print(row[1]['country'], row[1]['cluster'])
