import json

with open('D:/data center/hw3/pubs_validate.json', 'r') as f:
    data = json.load(f)


def spilt_with_org():
    result = {}
    for key in data.keys():
        org_list = {}
        for paper in data[key]:
            for person in paper['authors']:
                com_name = person['name'].lower().replace(' ', '_')
                if com_name == key:
                    if person['org'] in org_list:
                        org_list[person['org']].append(paper['id'])
                    else:
                        org_list[person['org']] = []
                        org_list[person['org']].append(paper['id'])
        result[key] = []
        for item in org_list.keys():
            result[key].append(org_list[item])

    with open('result8.json', 'w') as f:
        json.dump(result, f, indent=4)


def compare(authors_1, authors_2):
    names_1, orgs_1 = authors_1
    names_2, orgs_2 = authors_2
    if len(names_1.intersection(names_2)) >= 1.5 or len(orgs_1.intersection(orgs_2)) >= 0.5:
        return True
    else:
        return False


def compare_with_org():
    results = dict()
    for name, papers in data.items():
        papers_dic = {}
        paper_list = []
        for ix, paper in enumerate(papers):
            name_list = set()
            org_list = set()
            paper_id = paper['id']
            paper_list.append(paper_id)
            authors = paper['authors']
            for author in authors:
                com_name = author['name'].lower()
                name_list.add(com_name)
                org = author['org'].lower()
                if org != '':
                    org_list.add(org)
            papers_dic[paper_id] = (name_list, org_list)

        labels = [-1] * len(papers)
        paper_clustered = set()
        cluster_num = 0
        for idx_1 in range(len(papers)):
            paper_1 = papers[idx_1]
            paper_id_1 = paper_1['id']
            info_1 = papers_dic[paper_1['id']]

            for idx_2 in range(idx_1 + 1, len(papers)):
                paper_2 = papers[idx_2]
                pid_2 = paper_2['id']
                info_2 = papers_dic[paper_2['id']]
                if pid_2 in paper_clustered:
                    continue
                if compare(info_1, info_2):
                    if paper_id_1 not in paper_clustered:
                        paper_clustered.add(paper_id_1)
                        labels[idx_1] = cluster_num
                        cluster_num += 1
                    labels[idx_2] = labels[idx_1]
                    paper_clustered.add(pid_2)
        result = []
        for ix in range(len(set(labels)) + 1):
            result.append([])
        for ix, pid in enumerate(paper_list):
            cluster_id = labels[ix]
            if cluster_id == -1:
                result.append([pid])
                # result[cluster_id].append(pid)
            else:
                result[cluster_id].append(pid)
        results[name] = result

    with open('result8.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    # spilt_with_org()
    compare_with_org()
