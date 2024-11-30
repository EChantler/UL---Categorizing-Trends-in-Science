#%% imports
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from utilities import SimpleLogger


#%% parameters
file_name = '../data/pp-data-1732640534.779797.csv'

category_variance_threshold = 0.001
word_variance_threshold = 0.001

year = [year for year in range(1990, 2000)]
k=4

year = [year for year in range(2000, 2010)]
k=8

year = [year for year in range(2010, 2020)]
k=6

year = [year for year in range(2020, 2025)]
k = 5

# year = [2020]
# k = 6
logger = SimpleLogger(f'../output/sample/kmeans-clusters-k-{k}-year-{year}-{datetime.now()}.log')
logger.log("Start of KMeans clustering run")
logger.log("Start of KMeans clustering run with parameters: ")
logger.log(f"file_name: {file_name}")
logger.log(f"category_variance_threshold: {category_variance_threshold}")
logger.log(f"word_variance_threshold: {word_variance_threshold}")
logger.log(f"year: {year}")
logger.log(f"k: {k}")

#%% load data

df = pd.read_csv(file_name)

# %% slice by year and drop year column
df = df[df['year'].isin(year)]
df.reset_index(drop=True, inplace=True)
# scaler = StandardScaler()
# df['year'] = scaler.fit_transform(df[['year']])
df.drop('year', axis=1, inplace=True)

#%% 
print(df.head())
# %% one hot encoding of categories by exploding the categories column
df = df.assign(categories=df['categories'].str.split(' '))
df = df.explode('categories')
df['categories'] = df['categories'].str.split('.').str[0]

df = pd.get_dummies(df, columns=["categories"])
df = df.astype({col: 'int32' for col in df.columns if col.startswith('categories_')})

category_columns = [col for col in df.columns if col.startswith("categories_")]

logger.log(f"Categories ({len(category_columns)}): {category_columns}")

#%% feature variance threshold for categories
df_categories = df[category_columns]
category_selector = VarianceThreshold()
category_selector.fit_transform(df_categories)

category_variances = pd.DataFrame({'features': category_columns, 'variances': category_selector.variances_})
print(category_variances)
print(category_variances[category_variances['variances'] > category_variance_threshold])

kept_category_columns = [col for col, keep in zip(category_columns, category_selector.get_support()) if keep]

df = df.drop(columns=[col for col in category_columns if col not in kept_category_columns])
remaining_category_sums = df[kept_category_columns].sum(axis=1)
df = df[remaining_category_sums > 0]
df.reset_index(drop=True, inplace=True)

logger.log(f"Categories kept after VarianceThreshold ({len(kept_category_columns)}): {kept_category_columns}")

#%% updated stop words

vectorizer = TfidfVectorizer(max_features=100, 
                             lowercase=True,
                             max_df=0.8,
                             min_df=2,
                             stop_words='english', 
                             ngram_range=(2,3), 
                             smooth_idf=True
                            )
tfidf_title = vectorizer.fit_transform(df['title'])
word_feature_names = vectorizer.get_feature_names_out()

# Map indices back to feature names
word_columns = ["word_" + name for name in word_feature_names]
logger.log(f"Word features ({len(word_feature_names)}): {word_feature_names}")
tfidf_df = pd.DataFrame(tfidf_title.toarray(), columns=word_columns)

#drop title column
df = pd.concat([df, tfidf_df], axis=1)
del tfidf_title
del tfidf_df
df.drop('title', axis=1, inplace=True)

#%% feature variance threshold for title words
df_words = df[word_columns]
word_selector = VarianceThreshold(threshold=word_variance_threshold)
word_selector.fit_transform(df_words)

word_variances = pd.DataFrame({'features': word_columns, 'variances': word_selector.variances_})
print(word_variances)
print(word_variances[word_variances['variances'] > word_variance_threshold])

kept_word_columns = [col for col, keep in zip(word_columns, word_selector.get_support()) if keep]

logger.log(f"Words kept after VarianceThreshold ({len(kept_word_columns)}): {kept_word_columns}")

remaining_word_sums = df[kept_word_columns].sum(axis=1)

df = df.drop(columns=[col for col in word_columns if col not in kept_word_columns])
df = df[remaining_word_sums > 0]
df.reset_index(drop=True, inplace=True)

#%%
logger.log(f"Applying KMeans with k={k} to data with shape: {df.shape}")
model = KMeans(n_clusters=k, init='k-means++', max_iter=600, n_init=2)
cluster_labels = model.fit_predict(df)


centroids = model.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns=df.columns)
logger.log(f"Centroids: {centroid_df}")

# %%
# Visualize clusters
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
reduced_data = pca.fit_transform(df)
# Step 2: Scatter Plot with Cluster Labels
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=3, alpha=0.6)
plt.colorbar(label="Cluster Labels")
plt.title(f"PCA Visualization of KMeans Clusters for k: {k} Year: {year}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
# Add centroids
cluster_centers_reduced = pca.transform(centroids)
print(cluster_centers_reduced.shape)
plt.scatter(
    cluster_centers_reduced[:, 0], 
    cluster_centers_reduced[:, 1], 
    c='red', 
    s=200, 
    alpha=0.8, 
    marker='X', 
    label='Centroids'
)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=model.predict(df), cmap='viridis', s=5)
for i, (x, y) in enumerate(cluster_centers_reduced):
    plt.text(
        x, y, str(i), fontsize=12, color='white', 
        ha='center', va='center'
    )

plt.legend()
# plt.show()
plt.savefig(f"../output/k-means-clusters-k-{k}-year-{year}.png")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative Variance Explained: {pca.explained_variance_ratio_.cumsum()}")

#%% determine niche features per cluster
from collections import Counter
top_features = {}
for cluster_id in range(len(centroids)):
    top_features[cluster_id] = centroid_df.iloc[cluster_id].abs().nlargest(20).index.tolist()
print(top_features)
all_top_features = [feature for features in top_features.values() for feature in features]
print(all_top_features)
top_feature_counts = Counter(all_top_features)
niche_features = {}
for cluster_id, features in top_features.items():
    niche_features[cluster_id] = [feature for feature in features if top_feature_counts[feature] == 1]
print("Niche Features:")
print(niche_features)
# flattern niche features
niche_features = [feature for features in niche_features.values() for feature in features]
print(niche_features)

# %% print top features
feature_names = df.columns
for index, row in centroid_df.iterrows():
    logger.log(f"Cluster {index}:")
    # Extract feature names and their corresponding strengths and order by strength before printing. only top 20
    top_features = np.argsort(row, axis=0)[::-1][:20]  # Indices of top 20 features for each cluster
    for feature, strength in zip(feature_names[top_features], row[top_features]):
        if(feature in niche_features):
            logger.log(f" -> {feature}: {strength:.4f}")
        else:
            logger.log(f"  {feature}: {strength:.4f}")
    logger.log(" ")

 
# %% Samples per cluster
for cluster_id in range(len(centroids)):
    cluster_data = df[cluster_labels == cluster_id]
    logger.log(f"Cluster {cluster_id}: {len(cluster_data)} samples")

# %% Elbow method
visualizer = KElbowVisualizer(model, k=(2,10), timings=True)
visualizer.fit(df)        
visualizer.show(outpath=f"../output/k-means-elbow-k-{k}-year-{year}.png")

#%% Silhouette Score
s_score = silhouette_score(df, cluster_labels)
logger.log(f"Silhouette Score: {s_score}")

#%% Silhouette Visualizer
# visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
# visualizer.fit(df)        
# visualizer.show(outpath="../output/k-means-silhouette-k-{k}-year-{year}.png")
#%%

logger.log("End of KMeans clustering run")


# %%
