import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 读取 entities.dict，生成一个名称到ID的映射字典
def load_entities_dict(filepath):
    name_to_id = {}
    id_to_name = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entity_id, name = line.strip().split('\t')
            name_to_id[name] = int(entity_id)
            id_to_name[int(entity_id)] = name
    return name_to_id, id_to_name

# 读取关系文件并查找成份关系三元组
def load_triples(filepath, relation_name, name_to_id):
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            if relation == relation_name:
                triples.append((head, tail))  # 只记录头实体和尾实体
    return triples

# 根据实体名称找到对应的嵌入向量
def get_embedding(entity_name, name_to_id, embeddings):
    entity_id = name_to_id.get(entity_name, None)
    if entity_id is not None:
        return embeddings[entity_id]
    else:
        raise ValueError(f"Entity '{entity_name}' not found in entities.dict")

# 计算余弦相似度
def compute_cosine_similarity(embedding, target_embeddings):
    embedding = embedding.reshape(1, -1)  # 调整为二维向量
    return cosine_similarity(embedding, target_embeddings)

# 读取所有三元组文件并查找特定方剂的相关信息
def find_prescription_info(prescription_name, relation_list, name_to_id, id_to_name, triple_files):
    info = {relation: [] for relation in relation_list}
    prescription_id = name_to_id.get(prescription_name)

    if prescription_id is None:
        return info  # 如果方剂不在字典中，返回空信息

    for file in triple_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                head_id = name_to_id.get(head, None)
                if head_id == prescription_id and relation in relation_list:
                    info[relation].append(id_to_name.get(name_to_id.get(tail), tail))
    return info

# 加载所需数据
entities_dict_path = '../data/TCM/entities.dict'
relations_dict_path = '../data/TCM/relations.dict'
entity_embedding_path = '../models/Hake_TCM_0/entity_embedding.npy'
triple_files = ['../data/TCM/demo.txt']

# 加载实体映射和嵌入
name_to_id, id_to_name = load_entities_dict(entities_dict_path)
embeddings = np.load(entity_embedding_path)

# 输入的方剂名称
input_prescription = '复方丹参片'

# 获取输入方剂的嵌入
input_embedding = get_embedding(input_prescription, name_to_id, embeddings)

# 搜索含有成份关系的头实体（方剂名称）
relation_name = '成份'
all_heads = set()
for file in triple_files:
    triples = load_triples(file, relation_name, name_to_id)
    all_heads.update([head for head, _ in triples])  # 提取头实体

# 获取所有头实体的嵌入
head_embeddings = []
head_names = []
for head in all_heads:
    try:
        head_embeddings.append(get_embedding(head, name_to_id, embeddings))
        head_names.append(head)
    except ValueError:
        pass  # 如果实体不在字典中，则跳过

head_embeddings = np.array(head_embeddings)

# 计算余弦相似度
cosine_similarities = compute_cosine_similarity(input_embedding, head_embeddings)

# 输出相似度最高的方剂
similarity_scores = list(zip(head_names, cosine_similarities[0]))
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# 定义要查找的关系类型
relation_list = ['成份','治疗', '有效', '功用']

# 输出前五个最相似的方剂及其相关信息
for name, score in similarity_scores[:6]:
    print(f"Prescription: {name}, Cosine Similarity: {score}")
    # 查找'治疗'、'有效'、'功用'的相关信息
    prescription_info = find_prescription_info(name, relation_list, name_to_id, id_to_name, triple_files)
    for relation in relation_list:
        related_entities = prescription_info[relation]
        if related_entities:
            print(f"  {relation}: {', '.join(related_entities)}")
        else:
            print(f"  {relation}: No information found.")
    print()  # 空行分隔结果
