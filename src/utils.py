import tensorflow as tf
import json
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def metrics(y_pred, y_true):
    """metics for acc, f1, auc

    :param y_pred: array, scores
    :param y_true: array, 0 or 1
    :return:
    """
    y_pred_copy = y_pred.copy()
    y_true_copy = y_true.copy()
    neg_num = y_true.shape[0] - np.sum(y_true)
    split_value = np.sort(y_pred)[neg_num]
    ind_pos = y_pred >= split_value
    ind_neg = y_pred < split_value
    y_pred[ind_pos] = 1
    y_pred[ind_neg] = 0
    acc = np.sum(y_pred == y_true) / len(y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="binary")
    auc = roc_auc_score(y_true=y_true_copy, y_score=y_pred_copy, average="micro")

    return acc, f1, auc

def linear(x):
    return x

def act_func(mode):
    """set different activate function

    :param mode:
    :return:
    """
    if mode == "linear":
        return linear
    elif mode == "tanh":
        return tf.nn.tanh
    elif mode == "sigmoid":
        return tf.nn.sigmoid
    elif mode == "relu":
        return tf.nn.relu


def pad_seq(seq,  length):
    """pad 0 to the sequence

    :param seq: list,
    :param length: int,
    :return:
    """

    if len(seq) > length:
        #print(len(seq))
        seq = seq[:length]
    else:
        seq += (length - len(seq))*[0]

    return seq



def get_pre_train_emb(item_dict, entity_emb_filename):
    """

    :param item_dict: dict, old_id: new_id
    :param entity_emb_filename:
    :return:
    """
    f = open(entity_emb_filename)
    ent_emb = json.load(fp=f)["ent_embeddings"]
    item_emb = np.zeros([len(item_dict), len(ent_emb[0])])
    for old_id, new_id in item_dict.items():
        item_emb[new_id, :] = ent_emb[old_id]

    return item_emb
