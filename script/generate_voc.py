import cPickle
import csv
import os
import json
train_file = open("local_train_splitByUser")
test_file = open("local_test_splitByUser")
feature_info = json.load(open('feature_config.json', 'r'))
FEATURE_COUNT = feature_info['Num_feature_except_uid']
voc_list = feature_info['voc_list']
f_train = csv.reader(train_file, delimiter="\t")
f_test = csv.reader(test_file, delimiter="\t")
uid_dict = {}
feature_dict_list = []
header_file = csv.reader(open("data_header"), delimiter="\t")
header = next(header_file)
for i in range(FEATURE_COUNT):
    feature_dict_list.append({})

def load_dict(f_reader):
    
    for arr in f_reader:
        #arr = line.strip("\n").split("\t")
        clk = arr[0]
        uid = arr[1]
        if uid not in uid_dict:
            uid_dict[uid] = 0
        uid_dict[uid] += 1

        for i in range(FEATURE_COUNT):
            feature = arr[i+2]
            
            feature_list = arr[i+2+FEATURE_COUNT]
            
                
            if(len(feature) == 0 or len(feature_list) == 0):
                break
            if feature not in feature_dict_list[i]:
                feature_dict_list[i][feature] = 0
            feature_dict_list[i][feature] += 1
            for f in feature_list.split('_'):
                if f not in feature_dict_list[i]:
                    feature_dict_list[i][f] = 0
                feature_dict_list[i][f] += 1
        
        

load_dict(f_train)
load_dict(f_test)


sorted_uid_dict = sorted(uid_dict.iteritems(), key=lambda x:x[1], reverse=True)

sorted_feature_dict = [sorted(i.iteritems(), key=lambda x:x[1], reverse=True) for i in feature_dict_list]

uid_voc = {}
index = 0
for key, value in sorted_uid_dict:
    uid_voc[key] = index
    index += 1

final_voc_list = []
for i in range(FEATURE_COUNT):
    cur_voc = {}
    cur_voc["default"] = 0
    index = 1
    for key, value in sorted_feature_dict[i]:
        cur_voc[key] = index
        index += 1
    final_voc_list.append(cur_voc)



cPickle.dump(uid_voc, open(voc_list[0], "wb"))

for i in range(FEATURE_COUNT):
    print('finish')
    cPickle.dump(final_voc_list[i], open(voc_list[i+1], "wb"))
