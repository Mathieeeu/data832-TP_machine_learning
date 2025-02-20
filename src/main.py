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

def pipeline(data: pd.DataFrame, scaler = sk.preprocessing.StandardScaler(), n_components = 2, n_clusters = 3) -> pd.DataFrame:
    """"
    Preprocessing : standardscaler
    Decomposition : PCA
    Clustering : KMeans (3 clusters : help needed, no help needed, may need help)
    """
    data_clean = data.drop('country', axis=1)
    pipe = sk.pipeline.Pipeline([('scaler', scaler),
                                 ('pca', sk.decomposition.PCA(n_components)),
                                 ('kmeans', sk.cluster.KMeans(n_clusters))])
    pipe.fit(data_clean)

    # Predict the cluster for each data point
    res = pd.DataFrame({'country': data['country'], 'cluster': pipe.predict(data_clean)})
    return res


def best_hyperparamaters(data: pd.DataFrame) -> tuple:
    data_clean = data.drop('country', axis=1)
    
    param_grid = {
        'scaler': [sk.preprocessing.StandardScaler(), sk.preprocessing.MinMaxScaler()],
        'pca_n_components': [2, 3, 4],  # Tester plusieurs valeurs
    }
    
    best_score = -1
    best_result = None

    for params in sk.model_selection.ParameterGrid(param_grid):
        pipe = sk.pipeline.Pipeline([
            ('scaler', params['scaler']),
            ('pca', sk.decomposition.PCA(n_components=params['pca_n_components'])),
            ('kmeans',sk.cluster.KMeans(n_clusters=3))
        ])
        pipe.fit(data_clean)

        # Prédire et évaluer
        labels = pipe.named_steps['kmeans'].labels_
        score = sk.metrics.silhouette_score(data_clean, labels)

        if score > best_score:
            best_score = score
            best_result = (params['scaler'], params['pca_n_components'])
        

    return best_result



def plot_cluster_label_on_a_map_of_the_Earth(data: pd.DataFrame, colors = {0: '#beffc1', 1: '#aad0ff', 2: '#ff9a95', -1: '#aaaaaa'}) -> None:
    if not os.path.isfile("data/countries.geojson"):
        world = gpd.read_file("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson")
        world.to_file("data/countries.geojson")
    else:
        world = gpd.read_file("data/countries.geojson")

    world = world.merge(data, how='left', left_on='ADMIN', right_on='country')
    world['cluster'] = world['cluster'].fillna(-1)
    world['color'] = world['cluster'].map(colors)
    world.plot(color=world['color'], legend=True, figsize=(10, 8))
    plt.title('Clusters of countries')
    plt.show()

def plot_data_on_a_graph_with_the_axes_being_the_two_components_of_the_PCA_and_the_color_being_the_cluster(data: pd.DataFrame, colors = ['#beffc1', '#aad0ff', '#ff9a95']) -> None:
    data_clean = data.drop('country', axis=1)
    pipe = sk.pipeline.Pipeline([
        ('scaler', sk.preprocessing.StandardScaler()),
        ('pca', sk.decomposition.PCA(n_components=2)),
        ('kmeans', sk.cluster.KMeans(n_clusters=3, random_state=42))
    ])
    pipe.fit(data_clean)

    data_pca = pipe.named_steps['pca'].transform(data_clean)
    data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    data_pca['cluster'] = pipe.named_steps['kmeans'].labels_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data_pca, x='PC1', y='PC2', hue='cluster', palette=colors, s=100, alpha=0.7)
    plt.title('Data')
    plt.show()

def plot_la_valeur_de_chaque_variable_pour_chaque_classe(data: pd.DataFrame, colors = ['#beffc1', '#aad0ff', '#ff9a95']) -> None:

    data_clean = data.drop('country', axis=1)

    # Pipeline : StandardScaler, PCA et K-Means
    pipe = sk.pipeline.Pipeline([
        ('scaler', sk.preprocessing.StandardScaler()),
        ('pca', sk.decomposition.PCA(n_components=min(2, data_clean.shape[1]))),
        ('kmeans', sk.cluster.KMeans(n_clusters=3, random_state=42, n_init=10))
    ])
    pipe.fit(data_clean)

    # Ajouter les labels des clusters aux données
    data_clean['cluster'] = pipe.named_steps['kmeans'].labels_

    # Calcul de la moyenne par cluster (c'est le truc qu'on affichera)$
    mean_values = data_clean.groupby('cluster').mean()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()

    # on trace chaque feature sur les subplots
    for i, feature in enumerate(mean_values.columns):
        for j in range(3):
            axes[i].bar(j, mean_values[feature][j], color=colors[j], width=0.6)
        axes[i].set_title(feature)
        axes[i].set_xticks([0, 1, 2])
        axes[i].set_xticklabels(["Cluster 0", "Cluster 1", "Cluster 2"])
        axes[i].set_ylabel("Valeur moyenne")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    fig.legend(handles, [f'Cluster {i}' for i in range(3)], loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustement pour la légende
    plt.show()


if __name__ == '__main__':

    np.random.seed(42)

    data = load_data(filename)
    features = mean_and_variance_for_each_feature(data)
    #correlation_matrix(data)

    # # plot_data(data)
    # plot_feature(data, feature='gdpp', nb_countries=10, ascending=False)
    # plot_feature(data, feature='child_mort', nb_countries=10, ascending=True)
    # plot_feature(data, feature='income', nb_countries=10, ascending=False)
    
    
    clustered_data = pipeline(data)
    # plot_cluster_label_on_a_map_of_the_Earth(clustered_data)

    scaler, nb_components = best_hyperparamaters(data)
    best_clustered_data = pipeline(data, scaler=scaler, n_components=nb_components)

    # plotage des visualisations 
    plot_data_on_a_graph_with_the_axes_being_the_two_components_of_the_PCA_and_the_color_being_the_cluster(data)
    plot_la_valeur_de_chaque_variable_pour_chaque_classe(data)
    plot_cluster_label_on_a_map_of_the_Earth(best_clustered_data)

    # for row in clustered_data.iterrows():
    #     print(row[1]['country'], row[1]['cluster'])

