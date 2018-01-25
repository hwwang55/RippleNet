import sys
sys.path.append("../")
import tensorflow as tf
import utils

class KVMemoryNetwork(object):
    def __init__(self, user_clicks, KG, config):
        self.n_entity = config.n_entity   # number of  entity
        self.n_relation = config.n_relation  # number of relation
        self.n_entity_emb = config.n_entity_emb  # the dimension of the entity,
        self.n_relation_emb = config.n_relation_emb # the dimension of the realtion
        self.n_map_emb = config.n_map_emb  # the dimension after mapping entity to a new space
        self.n_memory = config.n_memory  # the size of memory
        self.n_hops = config.n_hops  # the number of hops
        self.max_grad_norm = config.max_grad_norm  # used for gradient clip
        self.act_func = utils.act_func(config.act_func)  # non-linear activate function after doing some matix multiply
        self.batch_size = config.batch_size
        self.lr = config.lr  # learning rate
        self.reg_kg = config.reg_kg
        self.reg_emb = config.reg_emb
        self.dtype = config.dtype  # data type, [tf.float16, tf.float32, tf.float64]
        self.kg_ratio = config.kg_ratio
        self.user_click_limit = config.user_click_limit  # int, use history clicks limit number
        self.entity_limit = config.entity_limit  # int, entity out degree limit
        self.is_clip_norm = config.is_clip_norm  # bool, if clip the grad
        self.update_rule = config.update_rule  # different method to train the model
        self.predict_mode = config.predict_mode  # different method to do the final prediction
        self.n_DCN_layer = config.n_DCN_layer  # when choose DCN to do the prediction, the number of DCN layers
        self.output_using_all_hops = config.output_using_all_hops  # bool, if using all hops o when do the prediction
        self.item_update_mode = config.item_update_mode  # different wat to update the item embedding
        self.model_log = config.model_log  # str, log_kv dir
        self.load_model = config.load_model  # bool, if load the model
        self.is_map_feature = config.is_map_feature
        self.is_use_relation = config.is_use_relation
        self.user_clicks = user_clicks  # dict
        self.KG = KG  # knowledge graphg
        self.build_model()

    def build_model(self):
        self.build_inputs()
        self.build_embeddings()
        # feature map matrix
        self.A = tf.get_variable(name="map_A", shape=[self.n_entity_emb, self.n_map_emb], dtype=self.dtype,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # prediction weight for different hops
        self.w_predict = []
        for i in range(self.n_hops + 1):
            self.w_predict.append(tf.get_variable(name="w_predict" + str(i), shape=[1],
                                             dtype=self.dtype,initializer=tf.constant_initializer(1.0)))
        # item retrival
        # (batch size, n_entity_emb)
        item_emb = tf.nn.embedding_lookup(self.entity_embedding, self.item)
        if self.is_map_feature:
            item_emb_map = self.act_func(tf.matmul(item_emb, self.A))
        else:
            item_emb_map = item_emb
        # user key retrieval
        h_emb_hops = []
        r_emb_hops = []
        t_emb_hops = []
        for i in range(self.n_hops):
            # (batch size, n_memory, n_entity_emb)
            h_emb_hops.append(tf.nn.embedding_lookup(self.entity_embedding, self.mem_h[i]))
            # (batch size, n_memory, n_relation_emb, n_relation_emb)
            if self.is_use_relation:
                r_emb_hops.append(tf.nn.embedding_lookup(self.relation_embedding, self.mem_r[i]))
            # (batch size, n_memory, n_entity_emb)
            t_emb_hops.append(tf.nn.embedding_lookup(self.entity_embedding, self.mem_t[i]))
        # two transformation matrix for when updating item embeddings
        self.o_map_mat = tf.get_variable("o_map_mat", shape=[self.n_map_emb, self.n_map_emb], dtype=self.dtype,
                                         initializer=tf.contrib.layers.xavier_initializer())
        self.item_map_mat = tf.get_variable("item_map_mat", shape=[self.n_map_emb, self.n_map_emb], dtype=self.dtype,
                                            initializer=tf.contrib.layers.xavier_initializer())
        # final output of the key_addressing
        o_list = self.key_addressing(item_emb_map, h_emb_hops, r_emb_hops, t_emb_hops)
        # make prediction
        self.o = o_list[-1]
        logits = tf.squeeze(self.make_prediction(item_emb_map, o_list))
        probs = tf.sigmoid(logits)
        self.probs = probs
        # add the loss function, cross entropy
        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logits))
        self.loss_layer(logits, h_emb_hops, t_emb_hops, r_emb_hops)
        # add metrics
        # here just a two-classification for movie recommendation
        self.predictions = tf.cast(probs > 0.5, tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int32), self.predictions), tf.float32))
        # add the optimizer op
        self.optimizer_layer()
        # to save the model information every fixed number training epochs
        self.latest_checkpoint = tf.train.latest_checkpoint(self.model_log)
        # add the saver
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def loss_layer(self, logits, h_emb_hops, t_emb_hops, r_emb_hops):
        # cross entropy loss
        xe_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=logits))
        # knowledge graph loss
        score_sum = 0
        for hop in range(self.n_hops):
            expand_h = tf.expand_dims(h_emb_hops[hop], axis=2)
            expand_t = tf.expand_dims(t_emb_hops[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(expand_h, r_emb_hops[hop]), expand_t))
            zero_mask = (self.mem_mask[hop] + 1e6) / 1e6
            score_sum +=  tf.reduce_mean(tf.sigmoid(hRt) * zero_mask, axis=[0,1])
        kg_loss = - self.reg_kg *  score_sum
        # embedding loss
        emb_reg_loss = 0
        for hop in range(self.n_hops):
            zero_mask = (self.mem_mask[hop] + 1e6) / 1e6
            emb_reg_loss += tf.reduce_mean(tf.reduce_sum(h_emb_hops[hop] * h_emb_hops[hop], axis=2) * zero_mask, axis=[0,1])
            emb_reg_loss += tf.reduce_mean(tf.reduce_sum(t_emb_hops[hop] * t_emb_hops[hop], axis=2) * zero_mask, axis=[0,1])
            emb_reg_loss += tf.reduce_mean(tf.reduce_sum(r_emb_hops[hop] * r_emb_hops[hop], axis=[2,3]) * zero_mask, axis=[0,1])
        emb_reg_loss = self.reg_emb * emb_reg_loss

        self.loss = xe_loss + kg_loss + emb_reg_loss

    def optimizer_layer(self):
        """add the optimizer
        """
        if self.update_rule == "adam":
            optimizer = tf.train.AdamOptimizer
        elif self.update_rule == "momentum":
            optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer
        elif self.update_rule == "sgd":
            optimizer = tf.train.GradientDescentOptimizer

        # self.train_op = optimizer(learning_rate=self.lr).minimize(self.loss)
        optimizer_op = optimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer_op.compute_gradients(self.loss))
        if self.is_clip_norm:
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, self.max_grad_norm)
                         for gradient in gradients]
        self.train_op = optimizer_op.apply_gradients(zip(gradients, variables))

    def next_batch(self, interactions, current_pos):
        """generate the next batch data

        :param interactions: list, element like [user, item, label]
        :param current_pos: int
        :returns  items:
                  labels:
                  initial_entity: list, length: batch_size
        """

        if current_pos + self.batch_size < len(interactions):
            next_records = interactions[current_pos: current_pos + self.batch_size]
            current_pos += self.batch_size
        else:
            next_records = interactions[current_pos:]
            current_pos = len(interactions)
        h_list = []
        t_list = []
        r_list = []
        items = []
        labels = []
        init_h = []
        init_t = []
        init_r = []
        for interact in next_records:
            user = interact[0]
            if user in self.user_clicks:
                user_click = self.user_clicks[user]  # the items set of the user history
            else:
                print("user %d not in the training dataset" % user)
                assert 0 == 1
                continue
            items.append(interact[1])
            labels.append(interact[2])

            degree = min(self.user_click_limit, len(user_click))
            user_click = user_click[:degree]
            init_h_tmp = []
            init_t_tmp = []
            init_r_tmp = []
            for click in user_click:  # item
                degree = max(1, int(min(self.entity_limit, len(self.KG[click][0])) * self.kg_ratio))
                init_h_tmp += degree * [click]
                init_t_tmp += self.KG[click][0][0:degree] # tail
                init_r_tmp += self.KG[click][1][0:degree]  # relation
            init_h.append(init_h_tmp)
            init_t.append(init_t_tmp)
            init_r.append(init_r_tmp)
        h_list.append(init_h)
        t_list.append(init_t)
        r_list.append(init_r)

        for _ in range(self.n_hops-1):
            next_h, next_t, next_r = self.get_next_entity(t_list[-1])
            h_list.append(next_h)
            t_list.append(next_t)
            r_list.append(next_r)

        # get the memory length
        mem_len_list = []
        for i in range(self.n_hops):  # n_hops
            mem_len_batch = []
            for j in range(len(h_list[0])):  # batch_size
                # padding
                mem_len_batch.append(min(len(h_list[i][j]), self.n_memory))
                h_list[i][j] = utils.pad_seq(h_list[i][j], self.n_memory)
                r_list[i][j] = utils.pad_seq(r_list[i][j], self.n_memory)
                t_list[i][j] = utils.pad_seq(t_list[i][j], self.n_memory)
            mem_len_list.append(mem_len_batch)


        return items, labels, h_list, t_list, r_list, mem_len_list, current_pos

    def get_next_entity(self, inner_entity):
        """get the next Mui

        From the movies that rated by the user, to extend.
        :param inner_entity: list, batch_size x n_memory, [t1, t2, t3, t4], batch_size x N
        :return next_entity:
                next_relation
        """

        next_h = []
        next_t = []
        next_r = []
        for user_entity in inner_entity:  # batch_size
            next_h_tmp = []
            next_t_tmp = []
            next_r_tmp = []
            for entity in user_entity:  #
                if entity in self.KG:
                    tmp_entity = self.KG[entity]
                    degree = min(self.entity_limit, len(tmp_entity[0]))
                    next_h_tmp += degree * [entity]
                    next_t_tmp += tmp_entity[0][0:degree]  # entities
                    next_r_tmp += tmp_entity[1][0:degree]  # relations

            next_h.append(next_h_tmp)
            next_t.append(next_t_tmp)
            next_r.append(next_r_tmp)

        return next_h, next_t, next_r

    def build_inputs(self):
        with tf.name_scope("input"):
            self.mem_h = []
            self.mem_t = []
            self.mem_r = []
            self.mem_len = []
            self.mem_mask = []
            self.item = tf.placeholder(dtype=tf.int32, shape=[None], name="item")
            self.label = tf.placeholder(dtype=self.dtype, shape=[None], name="label")
            for hop in range(self.n_hops):
                self.mem_h.append(tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memory_head_" + str(hop)))
                self.mem_r.append(tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memory_relation_" + str(hop)))
                self.mem_t.append(tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memory_tail_" + str(hop)))
                self.mem_mask.append(tf.placeholder(dtype=self.dtype, shape=[None, self.n_memory], name="memory_mask_" + str(hop)))
                self.mem_len.append(tf.placeholder(dtype=tf.int32, shape=[None], name="memory_length_" + str(hop)))

    def build_embeddings(self):
        with tf.name_scope("embedding"):
            self.entity_embedding = tf.get_variable(name="entity", dtype=self.dtype,
                                                     shape=[self.n_entity, self.n_entity_emb],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            if self.is_use_relation:
                self.relation_embedding = tf.get_variable(name="relation", dtype=self.dtype,
                                                           shape=[self.n_relation, self.n_relation_emb,
                                                                  self.n_relation_emb],
                                                           initializer=tf.contrib.layers.xavier_initializer())

    def make_prediction(self, item_emb, o_list):
        """make prediction according to the item embedding and the variable o
        :return:
        """

        if self.predict_mode == "inner_product":
            return self.inner_product_predict(item_emb, o_list)
        elif self.predict_mode == "MLP":
            return self.MLP_predict(item_emb, o_list)
        elif self.predict_mode == "DCN":
            return self.DCN_predict(item_emb, o_list)

    def inner_product_predict(self, x, o_list):
        """compute  the probability according to the inner product
        :param x:
        :param y:
        :return:
        """

        y = o_list[-1] * self.w_predict[-1]
        if self.output_using_all_hops:
            for i in range(self.n_hops-1):
                y += o_list[i] * self.w_predict[i]
        logits = tf.reduce_sum(x * y, axis=1)
        return logits

    def MLP_predict(self, x, o_list):
        """
        :param x: [batch_size, n_entity_emb], variabale item_emb
        :param y: [batch_size, n_entity_emb], input o
        :return prob: [batch_size]
        """

        y = o_list[-1]
        # [batch_size, 2*n_entity_emb]
        out_feature = tf.concat([x, y], axis=0)
        w_pred = tf.get_variable("w_pred", shape=[2 * self.n_map_emb, 1], dtype=self.dtype,)
        b_pred = tf.get_variable("b_pred", shape=[1], dtype=self.dtype, initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(out_feature, w_pred) + b_pred

        return logits

    def DCN_predict(self, item_emb, o_list):
        """Deep & Cross network,
        ref: Deep & Cross Network for Ad Click Predictions
        :return:
        """

        if self.output_using_all_hops:
            x0 = tf.concat([item_emb] + o_list, axis=1)  # [batch_size, (n_hops + 1)*n_map_emb]
            x_dim = (self.n_hops + 1) * self.n_map_emb
        else:
            x0 = tf.concat([item_emb, o_list[-1]], axis=1)  #[batch_size, 2*n_map_emb]
            x_dim = 2 * self.n_map_emb
        x0 = tf.expand_dims(x0, axis=2)  # [batch_size, 2*n_map_emb, 1]
        output_layer = [x0]
        for i in range(self.n_DCN_layer):
            w = tf.get_variable("w_dcn_" + str(i), shape=[x_dim, 1],dtype=self.dtype, initializer=tf.contrib.layers.xavier_initializer())
            w_tile = tf.tile(tf.expand_dims(w, axis=0), multiples=[tf.shape(item_emb)[0], 1, 1])
            b = tf.get_variable("b_dcn_" + str(i), shape=[x_dim, 1], dtype=self.dtype, initializer=tf.constant_initializer(0.0))
            output_layer.append(tf.matmul(tf.matmul(x0, output_layer[-1], transpose_b=True), w_tile) + b + output_layer[-1])
        # final layer
        w = tf.get_variable("w_pred", shape=[x_dim, 1], dtype=self.dtype, initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(tf.squeeze(output_layer[-1]), w)
        return logits

    def key_addressing(self, item_emb_map, h_emb_hops, r_emb_hops, t_emb_hops):
        """

        :param item_emb_map: [batch_size, n_entity_emb]
        :param h_emp_hops: list, whose element shape [batch_size, n_memory, n_entity_emb]
        :param t_emp_hops: list,
        :return:
        """
        if self.is_map_feature:
            A = tf.tile(tf.expand_dims(self.A, axis=0), multiples=[tf.shape(item_emb_map)[0], 1, 1])
        o_list = []
        for hop in range(self.n_hops):
            if self.is_use_relation:
                #  h_emb_hops[hop]: [batch_size, n_memory, n_entity_emb] - > [batch_size, n_memory, n_entity, 1]
                expand_h = tf.expand_dims(h_emb_hops[hop], axis=3)
                # R: [batch_size, n_memory, n_relation_emb, n_relation_emb],
                # n_relation_emb = n_entity_emb
                # Rh : [batch_size, n_memory, n_entity_emb]
                Rh = tf.squeeze(tf.matmul(r_emb_hops[hop], expand_h), axis=3)
            else:
                Rh = h_emb_hops[hop]
            # if do the feature map or not, multiply A
            # if map, Rh_map : [batch_size, n_memory, n_map_emb]
            # if map, item_emb_map: [batch_size, n_map_emb]
            if self.is_map_feature:
                Rh_map = self.act_func(tf.matmul(Rh, A))
            else:
                Rh_map = Rh
            # Softmax
            expand_item_emb = tf.expand_dims(item_emb_map, axis=2)  # [batch_size, n_map_emb, 1]
            dotted = tf.squeeze(tf.matmul(Rh_map, expand_item_emb), axis=2)  # [batch_size, n_memory]
            # get off the effect of null paddings
            soft_dotted = dotted - self.mem_mask[hop]
            # calculate probabilities
            probs = tf.nn.softmax(soft_dotted)  # [batch_size, n_memory]
            # t: [batch_size, n_memory, n_entity_emb]
            # t_map: [batch_size, n_memory, n_map_emb]
            if self.is_map_feature:
                t_map = self.act_func(tf.matmul(t_emb_hops[hop], A))
            else:
                t_map = t_emb_hops[hop]
            # prob: [batch_size, n_memory] -> [batch_size, n_memory, 1]
            expand_probs = tf.expand_dims(probs, axis=2)
            # o: [batch_size, n_entity_emb]
            o = tf.reduce_sum(t_map * expand_probs, axis=1)
            # update the item embedding
            item_emb_map = self.update_item_embedding(item_emb_map, o)
            o_list.append(o)

        return o_list

    def update_item_embedding(self, item_emb, o):
        """update the item_emb

        different ways to update
        """

        if self.item_update_mode == "plus":
            item_emb = item_emb + o
        elif self.item_update_mode == "map_o":
            item_emb = item_emb + tf.matmul(o, self.o_map_mat)
        elif self.item_update_mode == "map_item":
            item_emb = tf.matmul(item_emb + o, self.item_map_mat)
        elif self.item_update_mode == "map_all":
            item_emb = tf.matmul(item_emb + tf.matmul(o, self.o_map_mat), self.item_map_mat)

        return item_emb







