import json
from gensim.models import word2vec

with open('D:/data center/hw3/pubs_validate.json', 'r') as f:
    data = json.load(f)

if __name__ == '__main__':
    sentences = list()
    for name, papers in data.items():
        for idx, paper in enumerate(papers):
            authors = paper['authors']
            for author in authors:
                name = author['name'].lower().replace(' ', '_')
                sentences.append(name)
                org = author['org']
                sentences.append(org.split())
            title = paper['title']
            sentences.append(title.split())
            if 'abstract' in paper:
                abstract = paper['abstract']
                sentences.append(abstract.split())
            keywords = paper['keywords']
            sentences.append(keywords)
            venue = paper['venue']
            sentences.append(venue.split())

    model = word2vec.Word2Vec(sentences, size=100, sg=1, min_count=1, window=5)
    model.save('word2vec')
