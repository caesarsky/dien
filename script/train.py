import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import csv
import json


f_feature = open('feature_config.json', 'r')
feature_info = json.load(f_feature)


query_list = feature_info['query_list']
item_feature = feature_info['item_feature_list']
FEATURE_COUNT = len(item_feature)
QUERY_COUNT = len(query_list)
voc_list = query_list + item_feature
voc_list = [(k+'_voc.pkl') for k in voc_list]
EMBEDDING_DIM = feature_info['Embedding_dim']
HIDDEN_SIZE = EMBEDDING_DIM * FEATURE_COUNT
ATTENTION_SIZE = EMBEDDING_DIM * FEATURE_COUNT
BATCH_SIZE = feature_info['batch_size']
MAXLEN = feature_info['max_len']
LEARNING_RATE_DECAY = feature_info['learning_rate_decay']
TEST_ITER = feature_info['test_iter']

best_auc = 0.0



train_auc_list = []
train_loss = []
train_accuracy = []
train_aux_loss = []
test_auc_list = []
test_loss_list = []
test_accuracy_list = []
test_aux_loss_list = []



def prepare_feature(input, i, maxlen = None, return_neg = False):

    lengths_x = [len(s[2][1]) for s in input]
    seqs = [inp[2][i] for inp in input]
    noclk_seqs = [inp[3][i] for inp in input]
    if maxlen is not None:
        new_seqs = []
        new_noclk_seqs = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs.append(inp[2][i][l_x - maxlen:])
                new_noclk_seqs.append(inp[3][i][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs.append(inp[2][i])
                new_noclk_seqs.append(inp[3][i])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs = new_seqs
        noclk_seqs = new_noclk_seqs
        if len(lengths_x) < 1:
            return None, None, None, None
    n_samples = len(seqs)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs[0][0])

    his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s, no_s] in enumerate(zip(seqs, noclk_seqs)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        his[idx, :lengths_x[idx]] = s
        noclk_his[idx, :lengths_x[idx], :] = no_s
    item = numpy.array([inp[1][i] for inp in input])
    if return_neg:
        return item, his, mid_mask, numpy.array(lengths_x), noclk_his
    else:
        return item, his, mid_mask, numpy.array(lengths_x)


def prepare_data(input, target, maxlen = None, return_neg = False):
    items = []
    his_list = []
    noclk_his_list = []
    mid_mask = []
    for i in range(FEATURE_COUNT):
        item, his, mid_mask, lengths_x, noclk_his = prepare_feature(input, i, maxlen, return_neg)
        items.append(item)
        his_list.append(his)
        noclk_his_list.append(noclk_his)
    query = []
    for i in range(QUERY_COUNT):
        query.append(numpy.array([inp[0][i] for inp in input]))
    
    if return_neg:
        return query, items, his_list, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_his_list

    else:
        return query, items, his_list, mid_mask, numpy.array(target), numpy.array(lengths_x)


def eval(sess, test_data, model, model_path):

    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, item, item_his, mid_mask, target, sl, noclk_his = prepare_data(src, tgt, return_neg=True)
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, item, item_his, mid_mask, target, sl, noclk_his])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    test_auc_list.append(test_auc)
    test_loss_list.append(loss_sum)
    test_accuracy_list.append(accuracy_sum)
    test_aux_loss_list.append(aux_loss_sum)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum

def train(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        batch_size = BATCH_SIZE,
        maxlen = MAXLEN,
        test_iter = TEST_ITER,
        save_iter = 100,
        model_type = 'DNN',
	seed = 2,
):
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file,FEATURE_COUNT, QUERY_COUNT, voc_list, batch_size, maxlen, shuffle_each_epoch=False)
        test_data = DataIterator(test_file,FEATURE_COUNT,QUERY_COUNT, voc_list, batch_size, maxlen)
        n_query, n = train_data.get_n()
        
        if model_type == 'DNN':
            model = Model_DNN(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n,n_query, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN': 
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        # model = Model_DNN(n_query, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        print('                                                                                      test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, best_model_path))
        sys.stdout.flush()

        
        
        iter = 0
        lr = 0.001
        
        for itr in range(3):
            print('iter start: ')
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            stored_arr = []
            for src, tgt in train_data:
                uids, item, item_his, mid_mask, target, sl, noclk_his = prepare_data(src, tgt, maxlen, return_neg=True)
                loss, acc, aux_loss = model.train(sess, [uids, item, item_his, mid_mask, target, sl, lr, noclk_his])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                
                prob, _, _, _ = model.calculate(sess, [uids,item, item_his, mid_mask, target, sl, noclk_his])
                prob_1 = prob[:, 0].tolist()
                target_1 = target[:, 0].tolist()
                for p ,t in zip(prob_1, target_1):
                    stored_arr.append([p, t])
                iter += 1
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    train_auc = calc_auc(stored_arr)
                    print('iter: %d ----> train_auc: %.4f ---- train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' % \
                                          (iter, train_auc, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    print('                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, best_model_path))
                    train_loss.append(loss_sum / test_iter)
                    train_accuracy.append(accuracy_sum / test_iter)
                    train_aux_loss.append(aux_loss_sum / test_iter)
                    
                    train_auc_list.append(train_auc)
                    stored_arr = []
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' %(iter))
                    model.save(sess, model_path+"--"+str(iter))
            lr *= LEARNING_RATE_DECAY
            print('iter end')
        
        
        

def test(
        train_file = "local_train_splitByUser",
        test_file = "local_test_splitByUser",
        batch_size = BATCH_SIZE,
        maxlen = MAXLEN,
        model_type = 'DNN',
	seed = 2
):

    model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, FEATURE_COUNT,QUERY_COUNT, voc_list, batch_size, maxlen)
        test_data = DataIterator(test_file, FEATURE_COUNT,QUERY_COUNT, voc_list, batch_size, maxlen)
        n_query, n = train_data.get_n()
        
        if model_type == 'DNN':
            model = Model_DNN(n,n_query, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
	        model = Model_WideDeep(n,n_query, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n,n_query,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n,n_query, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        with tf.summary.FileWriter('./test_log') as writer:
            writer.add_graph(sess.graph)
            model.restore(sess, model_path)
            print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, model_path))
            writer.flush()
        

if __name__ == '__main__':
    if len(sys.argv) == 4:
        SEED = int(sys.argv[3])
    else:
        SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if sys.argv[1] == 'train':
        train(model_type=sys.argv[2], seed=SEED)
        result = zip(train_auc_list, train_loss, train_accuracy, train_aux_loss, test_auc_list, test_loss_list, test_accuracy_list, test_aux_loss_list)
        with open('result_' + sys.argv[2] + '.csv', "w") as f:
            writer = csv.writer(f)
            for row in result:
                writer.writerow(row)
                
    elif sys.argv[1] == 'test':
        test(model_type=sys.argv[2], seed=SEED)
    else:
        print('do nothing...')

