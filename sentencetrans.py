from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = []
with open("cat_care.txt", "r") as file:
    cat_care_sentences = file.readlines()
    sentences.extend(cat_care_sentences)
with open("desk_make.txt", "r") as file:
    desk_make_sentences = file.readlines()
    sentences.extend(desk_make_sentences)
with open("bomb_make.txt", "r") as file:
    bomb_make_sentences = file.readlines()
    bomb_mean = np.mean(model.encode(bomb_make_sentences), axis=0)
    bomb_dists = [np.linalg.norm(model.encode(s)-bomb_mean) for s in bomb_make_sentences]
    bomb_dist_95 = np.percentile(bomb_dists, 100)
    sentences.extend(bomb_make_sentences)
with open("elec_make.txt", "r") as file:
    desk_make_sentences = file.readlines()
    sentences.extend(desk_make_sentences)

embeddings = model.encode(sentences)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)
for i, sentence in enumerate(sentences):
    x = embeddings_pca[i, 0]
    y = embeddings_pca[i, 1]
    plt.text(x, y, sentence)
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Embeddings')
plt.show()

for s in sentences:
    print(s, np.linalg.norm(bomb_mean - model.encode(s)) > bomb_dist_95)

n_samples = len(sentences)
distances = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

cos_sim = util.cos_sim(embeddings, embeddings)
import pandas as pd
df = pd.DataFrame(distances)
df.to_csv("st_distances.csv", index=False)
df = pd.DataFrame(cos_sim)
df.to_csv("st_cos_sim.csv", index=False)
