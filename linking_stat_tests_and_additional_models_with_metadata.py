import pandas as pd
import librosa
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from scipy.stats import f_oneway, permutation_test
import matplotlib.pyplot as plt
import os

# Step 1: Load metadata CSV
metadata_file = "C:\\Users\\johnp\\EE402Project\\Homework_5\\speakers_english.csv"
metadata_df = pd.read_csv(metadata_file)

# Step 2: Create a mapping from filenames to speaker IDs
filename_to_speakerid = {
    row['filename']: row['speakerid']
    for _, row in metadata_df.iterrows()
}

# Step 3: Function to extract MFCCs and other features
def extract_features_from_file(audio_file, frame_length=1):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta_delta = librosa.feature.delta(mfccs_delta)
    energy = librosa.feature.rms(y=y)[0]
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    f0 = np.mean(pitches, axis=1)

    num_frames = int(np.floor(len(y) / (frame_length * sr)))
    segment_mfccs, segment_f0, segment_energy = [], [], []

    for i in range(num_frames):
        start, end = i * frame_length * sr, (i + 1) * frame_length * sr
        if end <= len(y):
            mfcc_slice = mfccs[:, int(start):int(end)]
            f0_slice = f0[int(start):int(end)]
            energy_slice = energy[int(start):int(end)]
            if mfcc_slice.size > 0 and f0_slice.size > 0 and energy_slice.size > 0:
                segment_mfccs.append(np.nanmean(mfcc_slice, axis=1))
                segment_f0.append(np.nanmean(f0_slice))
                segment_energy.append(np.nanmean(energy_slice))

    if segment_mfccs and segment_f0 and segment_energy:
        return np.array(segment_mfccs), np.array(segment_f0), np.array(segment_energy)
    else:
        return np.array([]), np.array([]), np.array([])

# Step 4: Extract features from all audio files
audio_dir = "C:\\Users\\johnp\\EE402Project\\Homework_5\\English_Full"
audio_files = [f"english{i}.mp3" for i in range(1, 580)]

all_mfccs, all_f0, all_energy = [], [], []
birthplaces, speaker_ids = [], []

for audio_file in audio_files:
    file_path = os.path.join(audio_dir, audio_file)
    try:
        mfccs, f0, energy = extract_features_from_file(file_path)
        if mfccs.size > 0 and f0.size > 0 and energy.size > 0:
            filename_key = audio_file.replace(".mp3", "")
            if filename_key in filename_to_speakerid:
                speaker_id = filename_to_speakerid[filename_key]
                birthplace = metadata_df.loc[metadata_df['speakerid'] == speaker_id, 'birthplace'].values[0].lower()
                all_mfccs.append(mfccs)
                all_f0.append(f0)
                all_energy.append(energy)
                birthplaces.append(birthplace)
                speaker_ids.append(speaker_id)
            else:
                print(f"Warning: filename {audio_file} not found in metadata mapping.")
        else:
            print(f"Warning: Features extraction failed for {audio_file}")
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

print(f"Processed {len(all_mfccs)} audio files.")

# Step 5: Concatenate all features into a single array
all_mfccs = np.vstack(all_mfccs)
all_f0 = np.concatenate(all_f0)
all_energy = np.concatenate(all_energy)

# Step 6: Standardize the features
scaler = StandardScaler()
all_mfccs_scaled = scaler.fit_transform(all_mfccs)
all_f0_scaled = scaler.fit_transform(all_f0.reshape(-1, 1))
all_energy_scaled = scaler.fit_transform(all_energy.reshape(-1, 1))

# Step 7: Combine all features into one array
combined_features = np.hstack([all_mfccs_scaled, all_f0_scaled, all_energy_scaled])

# Step 8: PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(combined_features)

# Step 9: ANOVA for feature analysis
anova_results = []
for i in range(combined_features.shape[1]):
    groups = [combined_features[:, i][np.array(birthplaces) == group] for group in np.unique(birthplaces)]
    f_stat, p_value = f_oneway(*groups)
    anova_results.append((f_stat, p_value))
    print(f"Feature {i + 1}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4e}")

# Step 10: Clustering and silhouette analysis for optimal k
k_range = range(2, 11)
silhouettes = []
for k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(combined_features)
    silhouettes.append(silhouette_score(combined_features, kmeans.labels_))

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouettes, marker='o')
plt.title("Silhouette Scores for Different k")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.show()

best_k = 3  # Update based on silhouette analysis
print(f"Chosen number of clusters: {best_k}")

# Step 11: Apply KMeans, Agglomerative, and GMM clustering
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(combined_features)

agg_clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
agg_labels = agg_clustering.fit_predict(combined_features)

gmm = GaussianMixture(n_components=best_k, covariance_type='diag', random_state=42)
gmm_labels = gmm.fit_predict(combined_features)

# Step 12: Visualize clustering results using PCA-reduced features
for labels, method in zip([kmeans_labels, agg_labels, gmm_labels], ['KMeans', 'Agglomerative', 'GMM']):
    plt.figure(figsize=(10, 6))
    for cluster_id in range(best_k):
        cluster_points = reduced_features[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
    plt.title(f"{method} Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

# Step 13: Permutation tests for silhouette scores
def permutation_test_for_clustering(data, labels, n_permutations=1000):
    original_score = silhouette_score(data, labels)
    permuted_scores = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        permuted_score = silhouette_score(data, permuted_labels)
        permuted_scores.append(permuted_score)
    p_value = np.mean([score >= original_score for score in permuted_scores])
    return original_score, p_value

kmeans_score, kmeans_p = permutation_test_for_clustering(combined_features, kmeans_labels)
agg_score, agg_p = permutation_test_for_clustering(combined_features, agg_labels)
gmm_score, gmm_p = permutation_test_for_clustering(combined_features, gmm_labels)

print(f"KMeans: Silhouette Score = {kmeans_score:.4f}, p-value = {kmeans_p:.4e}")
print(f"Agglomerative Clustering: Silhouette Score = {agg_score:.4f}, p-value = {agg_p:.4e}")
print(f"GMM: Silhouette Score = {gmm_score:.4f}, p-value = {gmm_p:.4e}")

# Step 14: Random clustering comparison
def random_clustering_comparison(data, labels, n_permutations=1000):
    original_score = silhouette_score(data, labels)
    random_scores = []
    for _ in range(n_permutations):
        random_labels = np.random.randint(0, len(labels), len(labels))
        score = silhouette_score(data, random_labels)
        random_scores.append(score)
    mean_random_score = np.mean(random_scores)
    p_value = np.mean([score >= original_score for score in random_scores])
    return original_score, mean_random_score, p_value

kmeans_random_result = random_clustering_comparison(combined_features, kmeans_labels)
agg_random_result = random_clustering_comparison(combined_features, agg_labels)
gmm_random_result = random_clustering_comparison(combined_features, gmm_labels)

print(f"KMeans Random Comparison: Original Score = {kmeans_random_result[0]:.4f}, Mean Random Score = {kmeans_random_result[1]:.4f}, p-value = {kmeans_random_result[2]:.4e}")
print(f"Agglomerative Random Comparison: Original Score = {agg_random_result[0]:.4f}, Mean Random Score = {agg_random_result[1]:.4f}, p-value = {agg_random_result[2]:.4e}")
print(f"GMM Random Comparison: Original Score = {gmm_random_result[0]:.4f}, Mean Random Score = {gmm_random_result[1]:.4f}, p-value = {gmm_random_result[2]:.4e}")