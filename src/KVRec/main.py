import sys
sys.path.append("../")
import os
import tensorflow as tf
import Config
from kv_rec import KVMemoryNetwork
import pickle
import numpy as np
from sklearn.utils import shuffle
import utils
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def construct_feeds(model, items, labels, h_list, t_list, r_list, mem_len_list):
    """construct the feed dicts for model.

    h, t, r, padding 0 when the actual memory size less than n_memory
    :param model: KVMemoryNetwork
    :param items: list, shape: batch_size
    :param labels: list, for recommendation, whose element should be 0 or 1
    :param h_list: list, head entities, shape: n_hops x batch_size x n_memory
    :param t_list: list, tail entities, shape: n_hops x batch_size x n_memory
    :param r_list: list, ralation, shape: n_hops x batch_size x n_memory
    :param mem_len_list: list, the actual length of each memory, shape: n_hops x batch_size. contructing masks for softmax computing
    :return feed_dict:
    """
    feed_dict = {}
    feed_dict[model.item] = items
    feed_dict[model.label] = labels
    mem_mask_list = []
    for i in range(model.n_hops):
        mem_mask = np.zeros([len(items), model.n_memory])
        for j in range(mem_mask.shape[0]):
            mem_mask[j, mem_len_list[i][j]:] = -1e6
        mem_mask_list.append(mem_mask)

    for i in range(model.n_hops):
        assert np.array(h_list[i]).shape[1] == config.n_memory
        assert np.array(h_list[i]).shape[0] == np.array(mem_len_list[i]).shape[0]
        feed_dict[model.mem_h[i]] = np.array(h_list[i])
        feed_dict[model.mem_r[i]] = np.array(r_list[i])
        feed_dict[model.mem_t[i]] = np.array(t_list[i])
        feed_dict[model.mem_len[i]] = np.array(mem_len_list[i])
        feed_dict[model.mem_mask[i]] = np.array(mem_mask_list[i])

    return feed_dict

def parse_args():
    """
    parse the args
    :return:
    """
    parser = argparse.ArgumentParser(description="Run KeyValueRecSys")
    parser.add_argument("--task", type=str, default="book")
    parser.add_argument("--is_map_feature", type=bool, default=False)
    parser.add_argument("--item_update_mode", type=str, default="map_item")
    parser.add_argument("--act_func", type=str, default="linear")
    parser.add_argument("--kg_ratio", type=float, default=1.0)
    parser.add_argument("--reg_kg", type=float, default=0.1)
    parser.add_argument("--n_hops", type=int, default=2)
    parser.add_argument("--n_entity_emb", type=int, default=50)
    return parser.parse_args()

args = parse_args()
config = Config.Config(args.task, args.n_entity_emb)
config.n_hops = args.n_hops
config.reg_kg = args.reg_kg
config.kg_ratio = args.kg_ratio
config.is_map_feature = args.is_map_feature
config.item_update_mode = args.item_update_mode
config.act_func = args.act_func
n_user_news = 500
print("--------------------------------------------------------------------------")
print("task: " + config.task)
print("is_map_feature: " + str(config.is_map_feature))
print("act function: " + config.act_func)
print("item update mode: " + config.item_update_mode)
print("kg_ratio: " + str(config.kg_ratio))
print("reg_kg: " + str(config.reg_kg))
print("user click ratio: " + str(int(config.user_click_limit * config.kg_ratio)))
print("n_hops: " + str(config.n_hops))
print("n_entity_emb: " + str(config.n_entity_emb))
print("entity_limit: " + str(config.entity_limit))
print("user_click_limit: " + str(config.user_click_limit))
print("n_memory: " + str(config.n_memory))
sys.stdout.flush()
# Read the data
with open(config.data_filename, "rb") as f:
    train_data, test_data, user_dict, kg_dict = pickle.load(f)

model = KVMemoryNetwork(user_clicks=user_dict, KG=kg_dict, config=config)
# initialize the sessions
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
model.sess.run(init_op)
#  restore the model
#  check if there exists checkpoint, if true, load it
ckpt = tf.train.get_checkpoint_state(model.model_log)
if ckpt and ckpt.model_checkpoint_path and model.load_model:
    print("Load the checkpoint: %s" % ckpt.model_checkpoint_path)
    model.saver.restore(model.sess, ckpt.model_checkpoint_path)

for step in range(config.epochs):
    # shuffle the data
    train_data = shuffle(train_data, random_state=520)
    batch_iter = int(np.ceil(len(train_data) / model.batch_size))
    current_pos = 0
    for i in range(batch_iter):
        items, labels, h_list, t_list, r_list, mem_len_list, current_pos = model.next_batch(train_data, current_pos)
        feed_dict = construct_feeds(model, items, labels, h_list, t_list, r_list, mem_len_list)
        fetchs = [model.train_op, model.loss]
        _, loss = model.sess.run(fetchs, feed_dict)
    sys.stdout.flush()
    ##-------------------------------------------------------------------
    ###-----------------------Evaluation-------------------------------------------
    preds = []
    y_true = []
    batch_iter = int(np.ceil(len(test_data) / model.batch_size))
    current_pos = 0
    for i in range(batch_iter):
        items, labels, h_list, t_list, r_list, mem_len_list, current_pos = model.next_batch(test_data, current_pos)
        y_true += labels
        feed_dict = construct_feeds(model, items, labels, h_list, t_list, r_list, mem_len_list)
        fetchs = [model.probs, model.loss]
        probs, loss = model.sess.run(fetchs, feed_dict)
        preds.extend(list(probs))
    acc, f1, auc = utils.metrics(y_pred=np.array(preds), y_true=np.array(y_true))
    print("Evaluation step %d: acc: %f, f1: %f, auc: %f" % (step, acc, f1, auc))
    print("Evaluation ratio: %f" % (len(y_true) / len(test_data)))
    sys.stdout.flush()






