import numpy as np
from itertools import islice
from collections import OrderedDict
import os
import ast
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


# 定义一个函数，用于从文件中读取三元组
def read_triplets(file_path):
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            triplet = line.strip().split('\t')
            triplets.append(triplet)
    return triplets


# 读取entities.dict和relation.dict中的映射关系
def read_mapping_dict_name_to_id(file_path):
    mapping_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entity_id, entity_name = line.strip().split('\t')
            mapping_dict[entity_name] = int(entity_id)
    return mapping_dict


def read_mapping_dict_id_to_name():
    file_path = '../data/TCM/entities.dict'
    mapping_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entity_id, entity_name = line.strip().split('\t')
            mapping_dict[int(entity_id)] = entity_name
    return mapping_dict


def get_entity_relation_id( syndromes):

    # 读取entities.dict中的映射关系
    entities_dict_path = '../data/TCM/entities.dict'
    entities_dict = read_mapping_dict_name_to_id(entities_dict_path)

    # 读取relation.dict中的映射关系
    relation_dict_path = '../data/TCM/relations.dict'
    relation_dict = read_mapping_dict_name_to_id(relation_dict_path)

    # 合并所有的三元组
    # all_triplets = train_triplets + test_triplets + valid_triplets
    all_triplets = read_triplets('../data/TCM/demo.txt')

    # 筛选关系为“成份”的三元组
    ingredient_triplets = [triplet for triplet in all_triplets if triplet[1] == '成份']

    # 获取成份三元组中的头实体
    ingredient_heads = {triplet[0] for triplet in ingredient_triplets}


    filtered_triplets = ingredient_heads
    # 去除重复元素
    filtered_triplets = list(OrderedDict.fromkeys(filtered_triplets))

    # print(filtered_triplets)
    # formula_triplets = [triplet for triplet in all_triplets if triplet[1] == '成份']  # 获取方剂三元组
    formula_triplets = [triplet for triplet in all_triplets if triplet[1] == '治疗' or triplet[1] == '有效'or triplet[1] == '功用']  # 获取方剂三元组
    # 去除重复三元组
    unique_head_to_formula_triplets = {tuple(triplet) for triplet in formula_triplets}
    head_to_formula_triplets = [list(triplet) for triplet in unique_head_to_formula_triplets]
    # head_to_formula_triplets = [(id_, inner_list) for id_, inner_list in head_to_formula_triplets if inner_list]

    # print(head_to_formula_triplets)

    triplets_by_head = {}
    f_count = 0
    for head in filtered_triplets:
    # for head in islice(filtered_triplets,5000):
        formula = []
        for triple in head_to_formula_triplets:
            if head == triple[0]:
                # print(triple[2])
                formula.append(triple[2])
        print(head + ':')
        print(formula)
        f_count = f_count + 1
        print(str(len(filtered_triplets)) + '/' + str(f_count))
        if head not in triplets_by_head:
            triplets_by_head[head] = []
        if len(formula) > 0:
            formula.sort()
        triplets_by_head[head].append(formula)
    # print(triplets_by_head)

    # 使用字典组织三元组，键是头实体，将name转为id
    triplets_by_head_id = {}
    t_count = 0
    for triplet in triplets_by_head.items():
        print(triplet)
        t_count = t_count + 1
        print(str(len(triplets_by_head.items())) + '/' + str(t_count))
        head_entity_id = entities_dict.get(triplet[0])
        drug_ids = []
        for drug in triplet[1][0]:
            # print(drug)
            drug_id = entities_dict.get(drug)
            drug_ids.append(drug_id)

        if head_entity_id is not None and drug_ids is not None:
            if head_entity_id not in triplets_by_head_id:
                triplets_by_head_id[head_entity_id] = []
            # 将三元组转换为ID形式
            triplets_by_head_id[head_entity_id].append(drug_ids)
    # print(triplets_by_head_id)

    return triplets_by_head_id


def get_formula_treats_disease():


    # 读取entities.dict中的映射关系
    entities_dict_path = '../data/TCM/entities.dict'
    entities_dict = read_mapping_dict_name_to_id(entities_dict_path)

    # 读取relation.dict中的映射关系
    relation_dict_path = '../data/TCM/relations.dict'
    relation_dict = read_mapping_dict_name_to_id(relation_dict_path)

    # 合并所有的三元组
    # all_triplets = train_triplets + test_triplets + valid_triplets
    all_triplets = read_triplets('../data/TCM/demo.txt')

    # 筛选关系为“成份”的三元组
    filtered_triplets = [triplet for triplet in all_triplets if
                         triplet[1] == '治疗' or triplet[1] == '有效' or triplet[1] == '功用']  # or triplet[1] == '功用'
    return filtered_triplets


'''
调整矩阵行数
'''

def adjust_matrix_rows(matrix1, matrix2):
    # 调整矩阵的行数为较小的那个矩阵的行数
    min_rows = min(matrix1.shape[0], matrix2.shape[0])
    matrix1_adjusted = matrix1[:min_rows, :]
    matrix2_adjusted = matrix2[:min_rows, :]

    return matrix1_adjusted, matrix2_adjusted


def cosine_similarity_a(vector_a1, vector_b1):
    print(vector_b1)
    if len(vector_b1) <= 0:
        return -1
        # 计算 vector_a 和 vector_b 中的最大长度
    # 计算 vector_a 和 vector_b 中的最大长度
    vector_a = np.array(vector_a1)
    vector_b = np.array(vector_b1)
    # max_length = max(len(vector_a), len(vector_b))
    # 获取最大长度
    max_length = max(len(vector_a), len(vector_b))

    # 补零使两个向量长度相同
    vector_a_padded = np.pad(vector_a, (0, max_length - len(vector_a)))
    vector_b_padded = np.pad(vector_b, (0, max_length - len(vector_b)))


    # 将向量转换为二维数组
    vector_a_padded = vector_a_padded.reshape(1, -1)
    vector_b_padded = vector_b_padded.reshape(1, -1)
    average_similarity = cosine_similarity(vector_a_padded, vector_b_padded)[0, 0]

    return average_similarity


def convert_to_hashable(obj):
    # 如果是数组，逐个转换为元组中的元素
    if isinstance(obj, np.ndarray):
        return tuple(convert_to_hashable(item) for item in obj)
    # 如果是其他可哈希的类型，直接返回
    elif hash(obj):
        return obj
    # 对于不可哈希的类型，可以选择进行其他处理，例如转换为字符串
    else:
        return str(obj)


def jaccard_similarity_coefficient(vector1, vector2):
    # 将NumPy数组转换为元组
    tuple1 = convert_to_hashable(vector1)
    tuple2 = convert_to_hashable(vector2)

    # 创建集合
    set1 = set(tuple1)
    set2 = set(tuple2)

    # 计算交集和并集的大小
    intersection_size = len(set1.intersection(set2))
    # union_size = len(set1.union(set2))
    union_size = len(set1)

    # 计算杰卡德相似系数
    jaccard_similarity = intersection_size / union_size if union_size != 0 else 0.0

    print(jaccard_similarity)
    return jaccard_similarity


def get_embedding_from_npy(ids):
    file_path = '../models/Hake_TCM_0/entity_embedding.npy'
    loaded_array = np.load(file_path)
    # 提取对应ID的向量
    # print(ids)
    selected_vectors = loaded_array[ids]
    # print("加载的数组内容:")
    # print(len(loaded_array))
    # 对向量进行加和
    # sum_of_vectors = np.sum(selected_vectors, axis=0)
    #求平均
    # sum_of_vectors = np.mean(selected_vectors, axis=0).ravel(
    sum_of_vectors = np.mean(selected_vectors, axis=0)
    # print("加和后的向量:")
    # print(sum_of_vectors)
    return sum_of_vectors


def get_embedding_vector(ids):
    file_path = '../models/Hake_TCM_0/entity_embedding.npy'
    loaded_array = np.load(file_path)
    # 提取对应ID的向量
    # print(ids)
    selected_vectors = loaded_array[ids]
    return selected_vectors


if __name__ == '__main__':

    # input_syndrome = ['发热', '咽痛', '全身乏力', '鼻塞', '喷嚏', '酸痛', '头痛', '外感']
    input_syndrome = ['发热', '发热', '头晕', '咽痛']

    id_to_name = read_mapping_dict_id_to_name()
    triplets_by_head_id = get_entity_relation_id( input_syndrome)

    trail_id_vectors = []
    print(triplets_by_head_id.items())
    h_count = 0
    for head_entity_id, triplets_id in triplets_by_head_id.items():
    # for head_entity_id, triplets_id in islice(triplets_by_head_id.items(), 10000):
        h_count = h_count + 1
        print(f'头实体 ID "{head_entity_id}" 对应的三元组（ID形式）：')
        print(triplets_id[0])
        print(str(len(triplets_by_head_id.items())) + '/' + str(h_count))
        if len(triplets_id[0])>2:
              trail_sum_vectors = get_embedding_from_npy(triplets_id[0])
        else:
              trail_sum_vectors = []
        # trail_sum_vectors = get_embedding_from_npy(triplets_id[0])
        # trail_sum_vectors = get_embedding_vector(triplets_id[0])
        triplet_id = [head_entity_id, trail_sum_vectors]
        trail_id_vectors.append(triplet_id)
    # 读取entities.dict中的映射关系
    entities_dict_path = '../data/TCM/entities.dict'
    entities_dict = read_mapping_dict_name_to_id(entities_dict_path)
    # 存储找到的ID的数组
    found_ids = []
    input_syndrome.sort()
    # 查找每个实体对应的ID并存储到数组
    for entity in input_syndrome:
        entity_id = entities_dict.get(entity)
        if entity_id is not None:
            found_ids.append(entity_id)
        else:
            print(f'"{entity}" 不在entities.dict中')

    print(f'找到的实体ID数组: {found_ids}')
    print(len(found_ids))
    input_sum_vectors = get_embedding_from_npy(found_ids)
    # input_sum_vectors = get_embedding_vector(found_ids)
    head_id_similarity = []
    t_count = 0
    for trail in trail_id_vectors:
        # print(trail)
        # 计算余弦相似度
        similarity_score = cosine_similarity_a(input_sum_vectors, trail[1])
        # similarity_score = jaccard_similarity_coefficient(input_sum_vectors, trail[1])
        print(trail[0])
        print(f"余弦相似度: {similarity_score}")
        head_s = [trail[0], similarity_score]
        head_id_similarity.append(head_s)
        t_count = t_count + 1
        # print(trail[1])
        print(str(len(trail_id_vectors)) + '/' + str(t_count))

    # print(head_id_similarity)
    # 根据子数组的第二个元素降序排列
    sorted_array = sorted(head_id_similarity, key=lambda x: x[1], reverse=True)
    # print(sorted_array)
    name_similarity = []
    diseases = get_formula_treats_disease()
    for head in enumerate(sorted_array[:5]):
        print(head)
        name = id_to_name.get(head[1][0])
        print(name + ":")
        # 去除重复三元组
        unique_diseases_set = {tuple(disease) for disease in diseases}
        unique_diseases = [list(disease) for disease in unique_diseases_set]
        # print(unique_diseases)
        for disease in unique_diseases:
            if disease[0] == name:
                print('  ' + disease[2])

