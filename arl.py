from embeddings.ARL_Adv.utils import *

# ----------------------------------
# IMPORTANT: Enable v1 compatibility
# ----------------------------------
tf.compat.v1.disable_eager_execution()


class ARL:
    def __init__(self, args: dict):
        # negative sample num
        self.neg_num = args['ns']
        # learning rate
        self.lr = args['lr']

        # Replace the old tf.contrib.layers.xavier_initializer with a Keras initializer
        from tensorflow.keras.initializers import GlorotUniform
        self.x_init = GlorotUniform()

    def set_session(self, sess):
        self.sess = sess

    def build(self, w_embed, c_embed):
        """
        Main build function: sets up variables, placeholders, forward pass, and optimizer.
        """
        self.define_word_embed(w_embed)
        self.define_cluster_embed(c_embed)
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    @staticmethod
    def get_variable(name, init, train):
        """
        Minimal wrapper to create a tf.compat.v1 variable using a constant initializer.
        """
        # Switch from tf.get_variable to tf.compat.v1.get_variable
        var = tf.compat.v1.get_variable(
            name=name,
            shape=init.shape,
            initializer=tf.compat.v1.constant_initializer(init),
            trainable=train
        )
        return var

    def define_word_embed(self, init):
        """
        Create a trainable word embedding variable with a 'pad' row of zeros on top.
        """
        # pad row
        pad = tf.zeros(shape=(1, tf.shape(init)[1]), name='w_padding', dtype=tf.float32)

        # get_variable (trainable)
        emb = self.get_variable('w_embed', init, True)

        # Concat zero row to the top
        self.w_embed = tf.concat([pad, emb], axis=0, name='w_embed_concat')

    def define_cluster_embed(self, init):
        """
        Create a trainable cluster embedding variable.
        """
        self.c_embed = self.get_variable('c_embed', init, True)

    def define_inputs(self):
        """
        Define placeholders for training (pos/neg sequences).
        """
        # Switch to tf.compat.v1 placeholders
        self.is_train = tf.compat.v1.placeholder_with_default(True, (), name='is_train')

        # We'll store shapes in normal Python variables
        shape = (None, None)

        # placeholders for sequences
        self.p_seq = tf.compat.v1.placeholder(tf.int32, shape=shape, name='p_seq')
        self.n_seqs = [
            tf.compat.v1.placeholder(tf.int32, shape=shape, name=f'n_seq_{i}')
            for i in range(self.neg_num)
        ]

        # lookups
        with tf.name_scope('lookup_pos'):
            self.p_lk = tf.nn.embedding_lookup(self.w_embed, self.p_seq)
            self.p_rep = tf.reduce_mean(self.p_lk, axis=1, keepdims=False)

        with tf.name_scope('lookup_neg'):
            self.n_lks = [
                tf.nn.embedding_lookup(self.w_embed, n, name=f'n_rep_{i}')
                for i, n in enumerate(self.n_seqs)
            ]
            # Concatenate the averaged negative embeddings
            self.n_reps = tf.concat(
                [tf.reduce_mean(n_lk, axis=1, keepdims=False) for n_lk in self.n_lks],
                axis=0
            )

    def define_optimizer(self):
        """
        Create an Adam optimizer with exponential decay, then build the gradient update op.
        """
        global_step = tf.compat.v1.train.get_or_create_global_step()

        lr_decayed = tf.compat.v1.train.exponential_decay(
            learning_rate=self.lr,
            global_step=global_step,
            decay_steps=50,
            decay_rate=0.99,
            staircase=False
        )

        self.optimizer = tf.compat.v1.train.AdamOptimizer(lr_decayed)

        # compute gradients
        self.sim_gvs = self.optimizer.compute_gradients(self.sim_loss)

        # apply gradients
        self.sim_opt = self.optimizer.apply_gradients(self.sim_gvs, name='sim_op', global_step=global_step)

    def get_loss_with(self):
        """
        Compute cluster probabilities (pc_probs), as well as pairwise and pointwise loss components.
        """
        with tf.name_scope('get_loss'):
            c_emb = self.c_embed

            pos_d, neg_d = self.p_rep, self.n_reps
            pc_probs, pos_recon = self.get_c_probs_and_recon_d(
                pos_d, c_emb, name='recon_d'
            )

            # L2-normalize the relevant vectors
            pos_d, pos_recon, neg_d = l2_norm_tensors(pos_d, pos_recon, neg_d)

            # dot product of pos_d and pos_recon
            pos_dr_sim = inner_dot(pos_d, pos_recon, keepdims=True)

            # dot product of pos_d with each negative sample
            pos_neg_sim = tf.matmul(pos_d, tf.transpose(neg_d))

            # margin-based loss
            pairwise_margin = tf.maximum(0., 1. - pos_dr_sim + pos_neg_sim)
            pairwise = tf.reduce_mean(pairwise_margin)

            # average positive similarity
            pointwise = tf.reduce_mean(pos_dr_sim)

        return pc_probs, pairwise, pointwise

    def get_c_probs_and_recon_d(self, d_emb, c_emb, name):
        """
        For a given document embedding `d_emb`, compute the cluster probabilities
        and reconstruct the doc embedding by weighting the cluster embeddings.
        """
        with tf.name_scope(name):
            c_score = tf.matmul(d_emb, tf.transpose(c_emb))
            c_probs = tf.nn.softmax(c_score, axis=1)
            recon = tf.matmul(c_probs, c_emb)
        return c_probs, recon

    def forward(self):
        """
        Run the model forward pass and define the final similarity loss.
        """
        self.pc_probs, pairwise, pointwise = self.get_loss_with()
        J1 = pairwise - pointwise
        self.sim_loss = J1

    def get_fd_by_batch(self, p_batch, n_batches=None):
        # 1) Extract token sequences
        p_seq_list = [d.tokenids for d in p_batch]
        # 2) Pad the positive batch
        p_seq_padded = pad_batch(p_seq_list, dtype='int32')

        # 3) Build feed pairs
        pairs = [(self.p_seq, p_seq_padded)]

        # 4) For negative batches, also pad
        if n_batches is not None:
            for placeholder, neg_batch in zip(self.n_seqs, n_batches):
                n_seq_list = [d.tokenids for d in neg_batch]
                n_seq_padded = pad_batch(n_seq_list, dtype='int32')
                pairs.append((placeholder, n_seq_padded))

        return dict(pairs)

    def train_step(self, pos, negs):
        """
        Single training step on one batch of data (pos + negs).
        """
        fd = self.get_fd_by_batch(pos, negs)
        self.sess.run(self.sim_opt, feed_dict=fd)

    def predict(self, batch):
        """
        Run forward pass in inference mode (is_train=False),
        returning cluster assignments and document embeddings.
        """
        fd = self.get_fd_by_batch(batch)
        fd[self.is_train] = False
        # We fetch the cluster probabilities and the doc embeddings (p_rep)
        cluster_probs, doc_embeddings = self.sess.run(
            [self.pc_probs, self.p_rep],
            feed_dict=fd
        )
        # Convert cluster probabilities into the actual cluster labels
        clusters = np.argmax(cluster_probs, axis=1)
        return clusters, doc_embeddings

    def save(self, file):
        """
        Save model variables to a checkpoint.
        """
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, file)

    def load(self, file):
        """
        Restore model variables from a checkpoint.
        """
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, file)
