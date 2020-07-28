
import cPickle
import csv
import os
import json
train_file = open("local_train_splitByUser")
test_file = open("local_test_splitByUser")
feature_info = json.load(open('feature_config.json', 'r'))
FEATURE_COUNT = feature_info['Num_history_feature']
QUERY_COUNT = feature_info['Num_query_feature']
voc_list = feature_info['voc_list']
f_train = csv.reader(train_file, delimiter="\t")
f_test = csv.reader(test_file, delimiter="\t")

feature_dict_list = []
query_dict_list = []
header_file = csv.reader(open("data_header"), delimiter="\t")
header = next(header_file)

for i in range(FEATURE_COUNT):
    feature_dict_list.append({})
for i in range(QUERY_COUNT):
    query_dict_list.append({})
def load_dict(f_reader):
    
    for arr in f_reader:
        #arr = line.strip("\n").split("\t")
        clk = arr[0]

        for i in range(QUERY_COUNT):
            query = arr[i+1].decode('utf-8')
            if query not in query_dict_list[i]:
                query_dict_list[i][query] = 0
            query_dict_list[i][query] += 1
            
            


        for i in range(FEATURE_COUNT):
            feature = arr[i+1+QUERY_COUNT].decode('utf-8')
            
            feature_list = arr[i+1+QUERY_COUNT+FEATURE_COUNT]
            
                
            if(len(feature) == 0 or len(feature_list) == 0):
                break
            if feature not in feature_dict_list[i]:
                feature_dict_list[i][feature] = 0
            feature_dict_list[i][feature] += 1
            for f in feature_list.split('_'):
                f = f.decode('utf-8')
                if f not in feature_dict_list[i]:
                    feature_dict_list[i][f] = 0
                feature_dict_list[i][f] += 1
        
        

load_dict(f_train)
load_dict(f_test)

sorted_query_dict = [sorted(i.iteritems(), key=lambda x:x[1], reverse=True) for i in query_dict_list]
sorted_feature_dict = [sorted(i.iteritems(), key=lambda x:x[1], reverse=True) for i in feature_dict_list]

final_voc_list = []
for i in range(QUERY_COUNT):
    cur_voc = {}
    cur_voc["default"] = 0
    index = 1
    for key, value in sorted_query_dict[i]:
        cur_voc[key] = index
        index += 1
    final_voc_list.append(cur_voc)

for i in range(FEATURE_COUNT):
    cur_voc = {}
    cur_voc["default"] = 0
    index = 1
    for key, value in sorted_feature_dict[i]:
        cur_voc[key] = index
        index += 1
    final_voc_list.append(cur_voc)




for i in range(FEATURE_COUNT):
    cPickle.dump(final_voc_list[i], open(voc_list[i], "wb"))
