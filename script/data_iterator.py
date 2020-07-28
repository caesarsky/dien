import numpy
import json
import cPickle as pkl
import random

import gzip

import shuffle



def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(pkl.load(f))


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIterator:

    def __init__(self, source,
                 NUM_FEATURE,
                 NUM_QUERY,
                 voc_list,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None):
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in voc_list:
            
            self.source_dicts.append(load_dict(source_dict))
        self.num_feature = NUM_FEATURE
        self.num_query = NUM_QUERY
        f_meta = open("item-info", "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1:self.num_feature]
        self.meta_id_map ={}
        for key in meta_map:
            
            if key in self.source_dicts[self.num_query]:
                mid_idx = self.source_dicts[self.num_query][key]
            else:
                mid_idx = 0
            val = []
            for i in range(len(meta_map[key])):
                idx = 0
                cur_val = meta_map[key][i]
                if(cur_val in self.source_dicts[i+self.num_query+1]):
                    idx = self.source_dicts[i+self.num_query+1][cur_val]
                val.append(idx)
            
                
            self.meta_id_map[mid_idx] = val

        f_review = open("reviews-info", "r")
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[self.num_query]:
                tmp_idx = self.source_dicts[self.num_query][arr[1]]
            self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_query = []
        for i in range(self.num_query):
            self.n_query.append(len(self.source_dicts[i]))
        
        
        self.n = []
        for i in range(self.num_feature):
            self.n.append(len(self.source_dicts[i+self.num_query]))
        

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_query, self.n

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source= shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by  history behavior length
            if self.sort_by_length:
                his_length = numpy.array([len(s[1+self.num_query+self.num_feature].split("_")) for s in self.source_buffer])
                tidx = his_length.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                query = []
                for i in range(self.num_query):
                    query.append(self.source_dicts[i][ss[i+1]] if ss[i+1] in self.source_dicts[i] else 0)
                
                item = []
                for i in range(self.num_feature):
                    item.append(self.source_dicts[i+self.num_query][ss[i+self.num_query+1]] if ss[i+self.num_query+1] in self.source_dicts[i+self.num_query] else 0)
                    
                item_list = []
                for i in range(self.num_feature):
                    tmp = []
                    for fea in ss[1+self.num_query+self.num_feature+i].split("_"):
                        m = self.source_dicts[i+self.num_query][fea] if fea in self.source_dicts[i+1] else 0
                        tmp.append(m)
                    item_list.append(tmp)

                # read from source file and map to word index

                #if len(mid_list) > self.maxlen:
                #    continue
                if self.minlen != None:
                    if len(item_list[0]) <= self.minlen:
                        continue
                if self.skip_empty and (not item_list[0]):
                    continue

                
                noclk_dict = {}
                for i in range(self.num_feature):
                    noclk_dict[i] = []
                for pos_mid in item_list[0]:
                    

                    noclk_tmp = {}
                    for i in range(self.num_feature):
                        noclk_tmp[i] = []
                    
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random)-1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp[0].append(noclk_mid)

                        
                        for i in range(1, self.num_feature):
                            
                            noclk_tmp[i].append(self.meta_id_map[noclk_mid][i-1])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    for i in range(self.num_feature):
                        noclk_dict[i].append(noclk_tmp[i])
                
                noclk_list = []
                for i in range(self.num_feature):
                    noclk_list.append(noclk_dict[i])
                source.append([query, item,  item_list, noclk_list])
                
                target.append([float(ss[0]), 1-float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target


