import json
import numpy as np
from gensim.models import word2vec
from sklearn.cluster import DBSCAN

model = word2vec.Word2Vec.load('./word2vec')
with open('D:/data center/hw3/pubs_validate.json', 'r') as f:
    data = json.load(f)


def words2vec(words):
    # generate a vector for a single word
    if len(words) == 0:
        return None
    if len(words) == 1 and words[0] == '':
        return None
    features = [model[word] for word in words]
    features = np.array(features)
    vec = np.mean(features, axis=0)
    vec = vec.reshape(1, 100)
    return vec


def sen2vec(sentence):
    # generate a vector for a single sentence
    words = sentence.split()
    vec = words2vec(words)
    return vec


def authors2vec(authors):
    # generate a vector for a single author
    features = list()
    for author in authors:
        name = author['name'].lower().replace(' ', '_')
        name_vec = words2vec(name)
        if name_vec is not None:
            features.append(name_vec)
        org = author['org']
        org_vec = sen2vec(org)
        if org_vec is not None:
            features.append(org_vec)
    features = np.concatenate(features)
    vec = np.mean(features, axis=0)
    vec = vec.reshape(1, 100)
    return vec


def paper2vec(paper):
    # generate a vector for a single paper
    features = list()
    authors = paper['authors']
    authors_vec = authors2vec(authors)
    if authors_vec is not None:
        features.append(authors_vec)

    title = paper['title']
    title_vec = sen2vec(title)
    if title_vec is not None:
        features.append(title_vec)
    if 'abstract' in paper:
        abstract = paper['abstract']
        abstract_vec = sen2vec(abstract)
        if abstract_vec is not None:
            features.append(abstract_vec)
    keywords = paper['keywords']
    keywords_vec = words2vec(keywords)
    if keywords_vec is not None:
        features.append(keywords_vec)
    venue = paper['venue']
    venue_vec = sen2vec(venue)
    if venue_vec is not None:
        features.append(venue_vec)

    features = np.concatenate(features)
    vec = np.mean(features, axis=0)
    vec = vec.reshape(1, 100)
    return vec


if __name__ == '__main__':
    results = dict()
    for name, papers in data.items():
        pid_list = []
        vectors = []
        for ix, paper in enumerate(papers):
            paper_vec = paper2vec(paper)
            vectors.append(paper_vec)
            pid = paper['id']
            pid_list.append(pid)
        vectors = np.concatenate(vectors)

        db = DBSCAN(eps=0.3, min_samples=3).fit(vectors)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # generate results
        result = []
        for idx in range(n_clusters):
            result.append([])
        for idx, pid in enumerate(pid_list):
            cluster_id = labels[idx]
            result[cluster_id].append(pid)
        print('length of result: {}'.format(len(result)))
        results[name] = result

    with open('result9.json', 'w') as f:
        json.dump(results, f, indent=4)
