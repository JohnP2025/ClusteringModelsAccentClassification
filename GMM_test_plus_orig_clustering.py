import pandas as pd
import librosa
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Step 1: Load metadata CSV
metadata_file = "C:\\Users\\johnp\\EE402Project\\Homework_5\\speakers_english.csv"
metadata_df = pd.read_csv(metadata_file)

# Step 2: Create a mapping from filenames to speaker IDs
filename_to_speakerid = {
    row['filename']: row['speakerid']
    for _, row in metadata_df.iterrows()
}

# Step 3: Process the audio files and extract MFCCs
def extract_mfcc_from_file(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)  # Aggregate MFCCs (mean across time)

# Step 4: Extract MFCC features from all audio files
audio_dir = "C:\\Users\\johnp\\EE402Project\\Homework_5\\English_Full"
audio_files = [f"english{i}.mp3" for i in range(1, 580)]

all_mfccs = []
birthplaces = []
speaker_ids = []

for audio_file in audio_files:
    file_path = os.path.join(audio_dir, audio_file)
    try:
        mfccs = extract_mfcc_from_file(file_path)
        filename_key = audio_file.replace(".mp3", "")
        if filename_key in filename_to_speakerid:
            speaker_id = filename_to_speakerid[filename_key]
        else:
            print(f"Warning: filename {audio_file} not found in metadata mapping.")
            continue

        birthplace = metadata_df.loc[metadata_df['speakerid'] == speaker_id, 'birthplace'].values[0].lower()
        all_mfccs.append(mfccs)
        birthplaces.append(birthplace)
        speaker_ids.append(speaker_id)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Ensure alignment
print(f"Processed {len(all_mfccs)} audio files.")

# Step 5: Concatenate all MFCCs into a single array
all_mfccs = np.vstack(all_mfccs)

# Step 6: Standardize the MFCCs
scaler = StandardScaler()
all_mfccs_scaled = scaler.fit_transform(all_mfccs)

# Step 7: Optimal clusters using silhouette scores
k_range = range(2, 11)
silhouettes = []

for k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_mfccs_scaled)
    silhouettes.append(silhouette_score(all_mfccs_scaled, kmeans.labels_))

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouettes, marker='o')
plt.title("Silhouette Scores for Different k")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.show()

best_k = 3  # Update based on the silhouette analysis
print(f"Chosen number of clusters: {best_k}")

# Step 8: Apply KMeans clustering
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(all_mfccs_scaled)

# Step 9: Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
agg_labels = agg_clustering.fit_predict(all_mfccs_scaled)

# Step 10: Apply Gaussian Mixture Models
gmm = GaussianMixture(n_components=best_k, covariance_type='diag', random_state=42)
gmm_labels = gmm.fit_predict(all_mfccs_scaled)

# Step 11: PCA for visualization
pca = PCA(n_components=2)
reduced_mfccs = pca.fit_transform(all_mfccs_scaled)

# Step 12: Visualize KMeans clustering results
plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_points = reduced_mfccs[kmeans_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
plt.title("KMeans Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# Step 13: Visualize Agglomerative Clustering results
plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_points = reduced_mfccs[agg_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
plt.title("Agglomerative Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# Step 14: Visualize GMM clustering results
plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_points = reduced_mfccs[gmm_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# Step 13: Silhouette scores for KMeans
kmeans_silhouette_score = silhouette_score(all_mfccs_scaled, kmeans_labels)
print(f"Silhouette Score for KMeans: {kmeans_silhouette_score}")

# Step 14: Silhouette scores for Agglomerative Clustering
agg_silhouette_score = silhouette_score(all_mfccs_scaled, agg_labels)
print(f"Silhouette Score for Agglomerative Clustering: {agg_silhouette_score}")

# Step 15: Silhouette scores for GMM
gmm_silhouette_score = silhouette_score(all_mfccs_scaled, gmm_labels)
print(f"Silhouette Score for GMM: {gmm_silhouette_score}")