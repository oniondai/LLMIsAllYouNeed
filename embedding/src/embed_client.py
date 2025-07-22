# encoding=utf-8
import requests
import time
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    
    # 计算点积
    dot_product = np.dot(a, b)
    
    # 计算范数
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # 避免除以零
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float("%.4f" % (dot_product / (norm_a * norm_b)))

def build_batch_query_similarity(embed_list):
    if not embed_list:
        return []
    
    sim_list = []
    for idx, _ in enumerate(embed_list):
        sim_list.append(cosine_similarity(embed_list[0], embed_list[idx]))
    
    return sim_list

def load_data():
    file = '/opt/meituan/dolphinfs_daicong/2.llm/data/测评数据.csv'
    query_list = []
    with open(file, 'r') as fin:
        for line in fin:
            items = line.strip().split(',')
            query_list.append(items)
    
    return query_list

query_list = [['观筵铁板烧&大渔旗下品牌','观筵TAIRYO', '山居满陇', 'PUTIEN_CN', 'FORNO'], ['池奈·咖喱蛋包饭','池奈curry', '王春春鸡汤饭', '皇甫壹号面倌', '耳之缘'],
              ['星巴克', 'Starbucks',   'RESTUCCO',   'CHICJOC', '瑞幸咖啡'],['敏华冰厅·鎏金食堂', 'MH-PRINT', 'Saboten_Wi-Fi', 'EmperorCinema-WiFi','北京鮨政']]

query_list = load_data()

def test_post():
    url = "http://0.0.0.0:8002/llm_embedding"
    start_time = time.time()
    for batch_query in query_list:
        request_data = {
            "queries": batch_query
        }

        response = requests.post(url, json=request_data)
        embed_list = response.json()
        print(embed_list)
        sim_list = build_batch_query_similarity(embed_list)
        print(batch_query, sim_list)

    print('------' * 10, "耗时：%.3fs" % (time.time()-start_time), '------' * 10)
   
if __name__ == '__main__':
    test_post()
