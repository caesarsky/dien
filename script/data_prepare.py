import csv
import os
import pandas as pd
import json

train_file = open("../train_sample.tsv")
test_file = open("../test_sample.tsv")
f_train = csv.reader(train_file, delimiter="\t")
f_test = csv.reader(test_file, delimiter="\t")

header_file = csv.reader(open("../pairwise_header.pointwise"), delimiter="\t")
header = next(header_file)

f_feature = open('feature_config.json', 'r')
feature_info = json.load(f_feature)

num_query = feature_info['Num_query_feature']
num_history = feature_info['Num_history_feature']

query_list = feature_info['query_list']
item_feature = feature_info['item_feature_list']
history_feature = feature_info['history_list']

column = query_list + item_feature + history_feature
c = query_list + item_feature + history_feature
c.insert(0, 'label')
item_dict = {}
review_dict = {"uuid": [], "wid_a": []}



def check_data(arr):
    
    for col in query_list:
        fea = arr[header.index(col)]
        if(len(fea) == 0) :
            return False
    for col in item_feature:
        fea = arr[header.index(col)]
        if(len(fea) == 0):
            return False
    
    for col in history_feature:
        fea = arr[header.index(col)]
        his_list = fea.strip("\n").split("_")
        if(len(fea) == 0 or len(his_list) < 5):
            return False
    return True

def load_dict(reader, path, data_type=""):
    idx = 0
    pos = 0
    data = {k: [] for k in column}
    data['label'] = []
    for arr in reader:
        if(check_data(arr) == False):
            continue
        
        label = arr[header.index("label_a")]
        if int(label) > 1:
            label = '1'
            pos+= 1
        else:
            label = '0'
        data['label'].append(str(label))
        
        for col in column:
            fea = arr[header.index(col)]
            data[col].append(fea)
        mid = arr[header.index(item_feature[0])]
        if mid not in item_dict:
            cur_dict = {}
            for i in range(1, len(item_feature)):
                col = item_feature[i]
                cur_dict[col] = arr[header.index(col)]
            item_dict[mid] = cur_dict
        
        review_dict["uuid"].append(arr[header.index("uuid")])
        review_dict["wid_a"].append(arr[header.index("wid_a")])
        idx+=1
        '''
        if(idx%5000 == 0):
            print(idx)
        '''
    df = pd.DataFrame(data)
    
    df.to_csv(path, sep="\t", header=False, index=False, columns=c)
    print(data_type + "dimention: " + str(df.shape))
    

print("data preparing ...")
load_dict(f_test, open("local_test_splitByUser","wb"),"testing ")

load_dict(f_train, open("local_train_splitByUser","wb"), "training ")
            
item_info = open("item-info", "w")
for key in item_dict:
    fea_list = []
    fea_dict = item_dict[key]
    for i in range(1, len(item_feature)):
        col = item_feature[i]
        fea_list.append(fea_dict[col])

    print >> item_info, key + "\t" + "\t".join(fea_list)
    
    

df_review = pd.DataFrame(review_dict)
df_review.to_csv(open("reviews-info", "wb"), sep="\t", header=False, index=False, columns=['uuid', 'wid_a'])
data_header = open("data_header", "wb")
header_output = csv.writer(data_header, delimiter="\t")
header_output.writerow(c)