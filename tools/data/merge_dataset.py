from difflib import SequenceMatcher
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import csv
import pandas as pd
import argparse
import glob
import os
import os.path as osp
import random

# 불용어(stopwords) 리스트 생성
stop_words = set(stopwords.words('english'))

# 두 개의 텍스트 파일에서 데이터를 읽어 리스트 생성하는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]
    
def similar(a, b):
    if round(float(SequenceMatcher(None, a, b).ratio()),2) * 100 > 30:
        words1 = a.split()
        words2 = b.split()

        # 불용어를 제외하여 단어가 최소한 3개 이상 일치할 경우 유사한 문장으로 판단
        filtered_words1 = [word for word in words1 if word.lower() not in stop_words]
        filtered_words2 = [word for word in words2 if word.lower() not in stop_words]

        return len(set(filtered_words1) & set(filtered_words2))
    return -1

def make_sinon_list(list1, list2):

    result_file = 'result.csv'

    # 유사한 문장의 인덱스를 저장할 리스트
    similar_indices = []

    # 첫 번째 리스트의 각 문장에 대해 두 번째 리스트에서 동일한 문장이 있는지 확인
    for idx2, sentence2 in enumerate(list2):
        temp = []
        for idx1, sentence1 in enumerate(list1):
            if sentence1 == sentence2:
                similar_indices.append(([idx1], idx2))
                print(f'{idx1}:{idx2}-{sentence1}:{sentence2}')
                break
            else:
                sim = similar(sentence1, sentence2)
                if sim >= 1:
                    if sim >= 2:
                        print(f'{idx1}:{idx2}-{sentence1}:{sentence2}')
                    temp.append(idx1)
        similar_indices.append((temp, idx2))

    with open(result_file, 'w', newline='') as csvfile:
        csv_header = ['kinetics400_idx', 'activity3D_idx', 'kinetics400_act', 'activity3D_act']
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        # 결과 출력 & CSV 저장
        if similar_indices:
            print("유사한 문장의 인덱스:")
            for idx1_list, idx2 in similar_indices:
                for idx1 in idx1_list:
                    print(f'{idx1}:{idx2}-{list1[idx1]}:{list2[idx2]}')
                    writer.writerow({'kinetics400_idx':f'{idx1}',
                                     'activity3D_idx':f'{idx2}',
                                     'kinetics400_act':f'{list1[idx1]}',
                                     'activity3D_act':f'{list2[idx2]}'})
        else:
            print("유사한 문장이 없습니다.")

def make_new_labels(file):
    type_data = pd.read_csv(file)
    # activity3D_idx 값을 기준으로 오름차순으로 정렬합니다.
    type_data = type_data.sort_values(by='activity3D_idx', ascending=True)
    
    add_count = 0
    n_kinetics = 400

    new_labels = []
    trans_labels = []

    for idx, row in type_data.iterrows():
        _type = row['type']
        _alabel = row['activity3D_act']
        _klabel = row['kinetics400_act']
        _aidx = row['activity3D_idx']
        _kidx = row['kinetics400_idx']

        if _type == 0:
            new_labels.append(_alabel)
            new_index = n_kinetics+add_count
            trans_labels.append((_aidx, new_index))
            # print(f'{_aidx} to {new_index}: {_alabel}')
            add_count = add_count + 1
        else:
            trans_labels.append((_aidx, _kidx))
            # print(f'{_aidx} to {_kidx}: {_klabel}')

    return trans_labels

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate new label list and Edit data annotation file')
    parser.add_argument('--src_label', type=str, default='tools/data/kinetics/label_map_k400.txt', help='source label list file')
    parser.add_argument('--tgt_label', type=str, default='tools/data/ETRI-Activity3D/label_map_ETRI-Activity3D.txt', help='target label list file')
    parser.add_argument('--output', type=str, default='data/k433/label_map_k433.txt', help='output label list file')
    parser.add_argument('--type_dir', type=str, default='./result_type.csv', help='label transition file')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # 첫 번째 텍스트 파일에서 데이터 읽어 리스트 생성
    # file_path_list1 = 'tools/data/kinetics/label_map_k400.txt'
    list1 = read_text_file(args.src_label)

    # 두 번째 텍스트 파일에서 데이터 읽어 리스트 생성
    # file_path_list2 = 'tools/data/ETRI-Activity3D/label_map_ETRI-Activity3D.txt'
    list2 = read_text_file(args.tgt_label)

    # # 비슷한 의미를 가진 문장을 찾는다
    # make_sinon_list(list1, list2)
    
    # 새로운 label을 생성한다
    # type_file = './result_type.csv'
    trans_labels = make_new_labels(args.type_dir)
    mappings = dict()

    for pair in trans_labels:
        ori_idx, new_idx = pair
        # print(f'{ori_idx} to {new_idx}')

        mappings[ori_idx] = new_idx

        if new_idx >= 400:
            list1.append(list2[ori_idx])
    
    # with open(args.output, 'w') as file:
    #     for item in list1:
    #         file.write(item + '\n')

    fileList1 = 'data/kinetics400/kinetics400_train_list_videos.txt'
    fileList2 = 'data/ETRI-Activity3D/Activity3D_train_list_videos.txt'

    newListFile = 'data/k433/k433_train_list_videos.txt'

    files1 = read_text_file(fileList1)
    files2 = read_text_file(fileList2)

    processed_lines = list()

    for line in files1:
        file_name, index = line.split()
        processed_lines.append(f'kinetics400_videos_train/{file_name} {index}\n')
    
    for line in files2:
        file_name, index = line.split()
        new_index = mappings.get(int(index))
        processed_lines.append(f'Activity3D_videos_train/{file_name} {new_index}\n')

    random.shuffle(processed_lines)

    with open(newListFile, 'w') as file:
        for items in processed_lines:
            file.write(items)
            
    