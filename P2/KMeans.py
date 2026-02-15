import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, num_centroids, max_iter=300, tol=1e-4):
        self.num_centroids = num_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.inertia_ = None

    def _initialize_centroids(self, data):
        """Inicializa los centroides usando el método k-means++."""
        centroids = [data[np.random.randint(0, len(data))]]
        for _ in range(1, self.num_centroids):
            distances = np.min(
                [np.linalg.norm(data - c, axis=1)**2 for c in centroids], axis=0
            )
            prob = distances / np.sum(distances)
            new_centroid = data[np.random.choice(range(len(data)), p=prob)]
            centroids.append(new_centroid)
        return np.array(centroids)

    def _calculate_inertia(self, data, labels):
        """Calcula la inercia (suma de distancias cuadradas a los centroides)."""
        inertia = 0
        for i in range(self.num_centroids):
            cluster_points = data[labels == i]
            inertia += np.sum(np.linalg.norm(cluster_points - self.centroids[i], axis=1)**2)
        return inertia

    def fit(self, data):
        """Entrena el modelo KMeans."""
        # Normalizar datos
        scaler = StandardScaler()

        # Inicializar centroides
        self.centroids = self._initialize_centroids(data)

        for _ in range(self.max_iter):
            # Asignar puntos al centroide más cercano
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Recalcular centroides
            new_centroids = np.array([
                data[labels == i].mean(axis=0) if len(data[labels == i]) > 0 else self.centroids[i]
                for i in range(self.num_centroids)
            ])

            # Verificar convergencia
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) <= self.tol):
                break

            self.centroids = new_centroids

        # Calcular inercia final
        self.inertia_ = self._calculate_inertia(data, labels)
        return labels

    def predict(self, data):
        """Clasifica nuevos datos en los clústeres aprendidos."""
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def plot_clusters(self, data, labels):
        """Visualiza los clústeres en 2D."""
        plt.figure(figsize=(8, 6))
        for i in range(self.num_centroids):
            cluster_points = data[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color="black", marker="x", s=100, label="Centroides")
        plt.legend()
        plt.title("Clusters y centroides")
        plt.show()
