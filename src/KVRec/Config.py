"""
Hyper-parameters configuration
"""
import tensorflow as tf

class Config(object):
    def __init__(self, task, n_entity_emb):
        self.task = task
        self.data_filename = {"book": "../../data/book/recommend.pkl"}[task]  # the serialized file store the preprocessing file
        self.n_entity = {"book": 79125}[task]  # number of entities
        self.n_memory = {"book": 459}[task]
        self.n_relation = {"book": 25}[task]  # number of relation
        self.entity_limit = {"book": 10}[task]
        self.user_click_limit = {"book": 30}[task]
        self.model_log = {"book": "../log_kv/book/"}[task]
        self.is_use_relation = True
        self.dtype = tf.float32
        self.batch_size = 128  # batch size for training# 64 ,128 for news
        self.n_entity_emb = n_entity_emb  # embedding size of entities (vector) and relations (2-D matrix)
        self.n_relation_emb = self.n_entity_emb
        self.n_hops = 2  # number of hops
        self.act_func = "linear"  # choice for nonlinear layer, ["linear", "tanh", "sigmoid", "relu"]
        self.item_update_mode = "map_item"  # update item mode, ["plus", "map_o",  "map_item", "map_all"]
        self.predict_mode = "inner_product"  # the method to compute the output final probabilities, ["MLP", "inner_product", "DCN"]
        self.n_DCN_layer = 2  # when user "DCN" do prediction, the layer number
        self.is_map_feature = False
        self.n_map_emb = 50  #the dimension of the feature after multiply a transformation matrix
        if self.is_map_feature is False:
            self.n_map_emb = self.n_entity_emb
        self.is_clip_norm = False  # decide if clip the grad norm
        self.max_grad_norm = 10  # maximum grad norm when doing clipping
        self.kg_ratio = 1.0  # the ratio of the kgs number of the first hop
        self.reg_kg = 0.01
        self.reg_emb = 1e-5  # l2 reg for embedding
        self.update_rule = "adam"  # update method when training the nn
        self.load_model = False  ## decide if load the model for continue training
        self.lr = 0.005  # learning rate
        self.epochs = 20  # total training epochs
        self.output_using_all_hops = True  # whether using the output of all hops to form user representation