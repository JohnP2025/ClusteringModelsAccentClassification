import pandas as pd
import librosa
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import silhouette_score
import os
import soundfile as sf


def spectral_gating_batch(input_dir, output_dir, sr=22050, threshold=0.02, win_length=441, hop_length=512):
    """
    Applies spectral gating to a batch of audio files in a directory and saves the denoised versions.

    Args:
        input_dir: Directory containing the input audio files.
        output_dir: Directory to save the denoised audio files.
        sr: Sampling rate for loading the audio files.
        threshold: Noise threshold for spectral gating.
        win_length: Window length for the STFT.
        hop_length: Hop length for the STFT.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # Only process audio files (e.g., wav, mp3)
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            input_file = os.path.join(input_dir, filename)

            # Load the audio file
            y, sr = librosa.load(input_file, sr=sr)

            # Apply spectral gating
            y_denoised = spectral_gating(y, sr, threshold, win_length, hop_length)

            # Save the denoised audio file
            output_file = os.path.join(output_dir, f"denoised_{filename}")
            sf.write(output_file, y_denoised, sr)
            print(f"Saved denoised file: {output_file}")

def spectral_gating(y, sr, threshold=0.02, win_length=441, hop_length=512):
    """
    Applies spectral gating to an audio signal.

    Args:
        y: The audio signal.
        sr: The sampling rate.
        threshold: The noise threshold.
        win_length: The window length for the STFT.
        hop_length: The hop length for the STFT.
    """

    # Compute the STFT
    D = librosa.stft(y, n_fft=win_length, hop_length=hop_length)

    # Compute the magnitude spectrogram
    mag = np.abs(D)

    # Estimate the noise floor
    noise_floor = np.percentile(mag, 25)

    # Create a binary mask
    mask = mag > (noise_floor + threshold)

    # Apply the mask to the spectrogram
    D_denoised = D * mask

    # Reconstruct the audio signal
    y_denoised = librosa.istft(D_denoised, hop_length=hop_length)

    return y_denoised





# Step 1: Load metadata CSV
metadata_file = "C:\\Users\\johnp\\EE402Project\\Homework_5\\speakers_english_denoised.csv"  # Replace with the actual path to your metadata file
metadata_df = pd.read_csv(metadata_file)

# Step 2: Create a mapping from filenames to speaker IDs
filename_to_speakerid = {}
for index, row in metadata_df.iterrows():
    filename = row['filename']
    speaker_id = row['speakerid']
    filename_to_speakerid[filename] = speaker_id

# Step 3: Process the audio files and extract MFCCs
def extract_mfcc_from_file(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)  # Aggregate MFCCs (mean across time)
    return mfccs_mean

# Step 4: Extract MFCC features from all audio files
orig_dir = "C:\\Users\\johnp\\EE402Project\\Homework_5\\English_Full"  # The directory where the audio files are located

# denoise audio files

audio_dir = "C:\\Users\\johnp\\EE402Project\\Homework_5\\English_Full_Denoised"
spectral_gating_batch(orig_dir, audio_dir)

audio_files = [f"denoised_english{i}.mp3" for i in range(1, 580)]  # Adjust according to your filenames

all_mfccs = []
birthplaces = []  # To store birthplace data
speaker_ids = []  # To store speaker ids for later referencing

for audio_file in audio_files:
    file_path = os.path.join(audio_dir, audio_file)
    try:
        # Extract MFCCs from the audio file
        mfccs = extract_mfcc_from_file(file_path)
        
        # Extract the speaker ID using the filename_to_speakerid mapping
        filename_key = audio_file.replace(".mp3", "")  # Remove the file extension
        if filename_key in filename_to_speakerid:
            speaker_id = filename_to_speakerid[filename_key]
        else:
            print(f"Warning: filename {audio_file} not found in metadata mapping.")
            continue  # Skip the file if there's no corresponding speaker ID
        
        # Extract birthplace from metadata
        birthplace = metadata_df.loc[metadata_df['speakerid'] == speaker_id, 'birthplace'].values[0]
        birthplace = birthplace.lower()  # Normalize birthplace text (optional)
        
        # Append the data
        all_mfccs.append(mfccs)  # Append MFCCs
        birthplaces.append(birthplace)  # Append birthplace info
        speaker_ids.append(speaker_id)  # Append speaker ID

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Ensure that both all_mfccs and birthplaces are aligned
print(f"Processed {len(all_mfccs)} audio files.")
print(f"Processed {len(birthplaces)} birthplaces.")

# Step 5: Concatenate all MFCCs into a single array (flattened across all files)
all_mfccs = np.vstack(all_mfccs)

# Step 6: Standardize the MFCCs
scaler = StandardScaler()
all_mfccs_scaled = scaler.fit_transform(all_mfccs)

# Step 7: Find the optimal number of clusters using the elbow method
distortions = []
silhouettes = []
k_range = range(2, 11)  # Try k between 2 and 10 clusters

for k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)  # Remove n_jobs and add n_init
    kmeans.fit(all_mfccs_scaled)
    distortions.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(all_mfccs_scaled, kmeans.labels_)
    silhouettes.append(silhouette_avg)

# Plot elbow method (distortion vs. number of clusters)
plt.figure(figsize=(10, 6))
plt.plot(k_range, distortions, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion (Inertia)")
plt.show()

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouettes, marker='o')
plt.title("Silhouette Scores for Different k")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Step 8: Choose the best k based on the elbow or silhouette method
best_k = 3  # You can adjust based on elbow method or silhouette score analysis
print(f"Chosen number of clusters: {best_k}")

# Step 9: Apply KMeans clustering with the chosen number of clusters
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(all_mfccs_scaled)

# Get cluster labels
labels = kmeans.labels_

# Step 10: Visualize the clusters (reduce dimensions for plotting)
pca = PCA(n_components=2)  # Reduce to 2D for visualization
reduced_mfccs = pca.fit_transform(all_mfccs_scaled)

# Plot the clusters
plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_points = reduced_mfccs[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
plt.title("KMeans Clustering of MFCCs from Multiple Audio Files")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# Step 11: Inspect cluster labels and metadata (birthplace)
# cluster_birthplaces = {i: [] for i in range(best_k)}

# Map birthplaces to clusters
# for i, label in enumerate(labels):
#     if i < len(birthplaces):  # Prevent index out of range error
#         cluster_birthplaces[label].append(birthplaces[i])

# # Print out the birthplaces associated with each cluster
# for cluster_id, birthplaces_in_cluster in cluster_birthplaces.items():
#     print(f"Cluster {cluster_id}: {Counter(birthplaces_in_cluster)}")

# Step 12: Apply Agglomerative Hierarchical Clustering
# Fit Agglomerative Clustering model on the same scaled features (all_mfccs_scaled)
agg_clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')  # 'ward' minimizes variance of clusters
agg_clustering_labels = agg_clustering.fit_predict(all_mfccs_scaled)

# Step 13: Visualize the Agglomerative Clustering results (using PCA for 2D visualization)
reduced_mfccs_agg = PCA(n_components=2).fit_transform(all_mfccs_scaled)  # Reduce dimensions for visualization

# Plot the Agglomerative Clustering results
plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_points = reduced_mfccs_agg[agg_clustering_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
plt.title("Agglomerative Clustering of MFCCs")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# Step 14: Optional - Compare Silhouette Score for Agglomerative Clustering
agg_silhouette_score = silhouette_score(all_mfccs_scaled, agg_clustering_labels)
print(f"Silhouette Score for Agglomerative Clustering: {agg_silhouette_score}")

# Optional: Inspect clusters' metadata (similar to what you did with KMeans)
# cluster_data_agg = {i: {"birthplaces": [], "audio_files": [], "speaker_ids": []} for i in range(best_k)}

# # Map birthplaces, audio files, and speaker IDs to Agglomerative Clustering clusters
# for i, label in enumerate(agg_clustering_labels):
#     if i < len(birthplaces):  # Prevent index out of range error
#         cluster_data_agg[label]["birthplaces"].append(birthplaces[i])
#         cluster_data_agg[label]["audio_files"].append(audio_files[i])
#         cluster_data_agg[label]["speaker_ids"].append(speaker_ids[i])

# # Print out the information associated with each cluster from Agglomerative Clustering
# for cluster_id, data in cluster_data_agg.items():
#     print(f"\nAgglomerative Cluster {cluster_id}:")
    
#     # Birthplaces count in the cluster
#     print(f"  Birthplaces: {Counter(data['birthplaces'])}")
    
#     # List audio files in this cluster
#     print(f"  Audio files in this cluster:")
#     for audio_file in data["audio_files"]:
#         print(f"    - {audio_file}")
        
#     # List speaker IDs (optional)
#     print(f"  Speaker IDs in this cluster:")
#     for speaker_id in data["speaker_ids"]:
#         print(f"    - {speaker_id}")