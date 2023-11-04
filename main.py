import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the number of clusters (K)
K = 3
TOTAL_NUMBER_OF_DATA = 150
centroids = []

clusters = []

def load_data():
    data = np.genfromtxt("iris.txt", delimiter=None, usecols=(0, 1, 2, 3))
    return data

class Centroid:
    def __init__(self, a, b, c, d):
        self.coordinates = np.array([a, b, c, d])
        self.count = 0

    def update(self, data_point):
        self.coordinates = (self.coordinates * self.count + data_point) / (self.count + 1)
        self.count += 1

    def reset(self):
        self.coordinates = np.array([0.0, 0.0, 0.0, 0.0])
        self.count = 0

def initialize_centroids():
    for k in range(K):
        c1 = Centroid(random.uniform(4, 8), random.uniform(1.5, 4.5), random.uniform(0.5, 7), random.uniform(0, 3))
        centroids.append(c1)
        clusters.append(0)

def get_distance(data_point, centroid):
    return np.linalg.norm(data_point - centroid.coordinates)

def assign_clusters(data, centroids):
    for i, data_point in enumerate(data):
        distances = [get_distance(data_point, centroid) for centroid in centroids]
        min_index = np.argmin(distances)
        data_point.set_cluster(min_index)

def update_centroids(data, centroids):
    for centroid in centroids:
        centroid.reset()

    for data_point in data:
        assigned_cluster = data_point.get_cluster()
        centroids[assigned_cluster].update(data_point.get_data())

def kmeans(data, K):
    initialize_centroids()
    data = [DataPoint(data[i]) for i in range(len(data))]
    for iteration in range(10):
        assign_clusters(data, centroids)
        update_centroids(data, centroids)

def show_results():
    miss_classified_data = 0
    for number in clusters:
        print(number, " data points belong to cluster number ", clusters.index(number) + 1)
        if number - 50 > 0:
            miss_classified_data += number - 50
    print('Accuracy: ', 100 - ((miss_classified_data / 150.0) * 100), '%')

def plot_clusters(data, centroids):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    for i in range(K):
        cluster_data = np.array([data_point.get_data() for data_point in data if data_point.get_cluster() == i])
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i+1}')
        centroid_coordinates = centroids[i].get_coordinates()
        plt.scatter(centroid_coordinates[0], centroid_coordinates[1], c='red', s=100, marker='x', label=f'Centroid {i+1}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right')
    plt.title('K-means Clustering')
    plt.show()

def plot_inertia(data, K_range):
    inertias = []
    for k in K_range:
        initialize_centroids()
        kmeans(data, k)
        inertia = 0
        for i, centroid in enumerate(centroids):
            cluster_data = np.array([data_point.get_data() for data_point in data if data_point.get_cluster() == i])
            inertia += np.sum((cluster_data - centroid.get_coordinates())**2)
        inertias.append(inertia)
    
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs. Number of Clusters')
    plt.show()

def save_clustered_data(data, clusters):
    clustered_data = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
    clustered_data['Cluster'] = clusters
    clustered_data.to_csv('clustered_data.csv', index=False)

def main():
    data = load_data()
    plot_inertia(data, range(1, 10))
    K = int(input('Enter the number of clusters (K): '))
    kmeans(data, K)
    show_results()
    plot_clusters(data, centroids)
    save_clustered_data(data, clusters)

if __name__ == "__main__":
    main()
