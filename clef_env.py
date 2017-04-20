# Environment for CLEF
import numpy as np
import tensorflow as tf
import os


class ClefEnv:
    NUM_TIME_SLOTS = 1000
    NUM_LEVELS = 4  # number of levels in one RLFD cycle
    # T is the period of a level in RLFD
    T_LIST = [1, 2, 3, 7, 10]
    P_LIST = [4. / 11, 3. / 11, 2. / 11, 1. / 11, 1. / 11]
    NEG_REWARD = -1 * max(T_LIST) * NUM_LEVELS
    NEG_REWARD_PROB = 1.0
    # MAX_T = 10 * NUM_LEVELS
    # NUM_COUNTER = 100
    # THRESHOLD = 1
    # EARDET_LIMIT = 100
    # OUTBOUND_CAPACITY = 10000
    ATK_STATUS_SIZE = 40
    TRAIN_CLEF = False
    POS_CLEF_REWARD = NUM_TIME_SLOTS

    # Actions:
    # 1 -> send traffic to the limit of EARDet, in a time slot
    # 0 -> send no traffic in a time slot
    ACTION_SPACE = [0, 1]

    def __init__(self, config=None):
        if config is not None:
            if "NUM_TIME_SLOTS" in config:
                self.NUM_TIME_SLOTS = config["NUM_TIME_SLOTS"]
            if "NUM_LEVELS" in config:
                self.NUM_LEVELS = config["NUM_LEVELS"]
            if "T_LIST" in config:
                self.T_LIST = config["T_LIST"]
            if "P_LIST" in config:
                self.P_LIST = config["P_LIST"]
            if "NEG_REWARD" in config:
                self.NEG_REWARD = config["NEG_REWARD"]
            if "NEG_REWARD_PROB" in config:
                self.NEG_REWARD_PROB = config["NEG_REWARD_PROB"]
            if "INPUT_SIZE" in config:
                self.ATK_STATUS_SIZE = config["INPUT_SIZE"]
                # TODO: the ATK_STATUS_SIZE maybe in other better value
            if "TRAIN_CLEF" in config:
                self.TRAIN_CLEF = config["TRAIN_CLEF"]
            if "POS_CLEF_REWARD" in config:
                self.POS_CLEF_REWARD = config["POS_CLEF_REWARD"]

            if len(self.P_LIST) != len(self.T_LIST):
                print "size of P_LIST != size of T_LIST!"
                exit()

        self.P_ACCU_LIST = list()
        p_accu = 0.
        for i in range(len(self.P_LIST)):
            p_accu += self.P_LIST[i]
            self.P_ACCU_LIST.append(p_accu)

        self.qNet = None
        if self.TRAIN_CLEF:
            print ">>>Enabled CLEF Training<<<"
            self.qNet = ClefQnet(
                self.ATK_STATUS_SIZE, len(self.T_LIST), config=config)

        self.reset()

    def reset(self):
        self.status = [0.5] * self.NUM_TIME_SLOTS
        self.next_idx = 0           # indicate the slot in timeline
        self.next_cycle_idx = 0     # indicate the slot in current cycle
        self.violate_amount = 0     # number of "true" slot so far
        self.cur_prob = list()
        self.cur_T = 1
        self.clef_cycle_reward = 0
        self.last_t = 0
        return self.status

    def get_action_space(self):
        return self.ACTION_SPACE

    def _pick_next_T_idx(self, t, status):
        if self.qNet is not None:
            return self.qNet.nextT_idx(t, status)
        else:
            return self._pick_T_randomly()

    def _pick_T_randomly(self):
        rand = np.random.rand()
        key_idx = len(self.P_ACCU_LIST) - 1
        for i in range(len(self.P_ACCU_LIST)):
            if rand < self.P_ACCU_LIST[i]:
                key_idx = i
                break
        return key_idx

    def step(self, action_idx):
        # s1, r, d, _ = env.step(a)
        self.last_action = self.ACTION_SPACE[action_idx]
        if self.next_cycle_idx == 0:
            self.last_t = self.next_idx
            # select a new cycle
            self.cur_prob = [0.] * self.NUM_LEVELS
            self.cur_T_idx = self._pick_next_T_idx(self.last_t, self.status)
            self.cur_T = self.T_LIST[self.cur_T_idx]
            self.clef_cycle_reward = 0  # the oversent traffic in new cycle

        r = 0
        d = False
        info = [False, False]   # [is end of level , is end of cycle]

        action = self.ACTION_SPACE[action_idx]
        level = self.next_cycle_idx / self.cur_T
        self.status[self.next_idx] = action
        if action == 1:
            # we have approximation over prob here
            self.cur_prob[level] = 1.
            self.violate_amount += 1
            r = 1

        self.next_cycle_idx += 1
        self.next_idx += 1
        self.clef_cycle_reward -= r

        if self.next_cycle_idx % self.cur_T == 0:
            info[0] = True

        if self.next_cycle_idx == self.NUM_LEVELS * self.cur_T:
            info[1] = True
            # if this is the last slot of this cycle
            # see whether we can detect the large flow
            prob_before_last_level = 1.
            for l in range(0, self.NUM_LEVELS - 1):
                prob_before_last_level *= self.cur_prob[l]
            detect_prob = prob_before_last_level * \
                self.cur_prob[self.NUM_LEVELS - 1]

            if np.random.rand(1) <= detect_prob:
                d = True
                if np.random.rand(1) <= self.NEG_REWARD_PROB:
                    r = self.NEG_REWARD
                    self.clef_cycle_reward += self.POS_CLEF_REWARD

            if self.qNet is not None:
                self.qNet.add_to_exp_buffer(
                    self.last_t, self.status, self.cur_T_idx,
                    self.clef_cycle_reward)

            # if prob_before_last_level == 1 and action == 0:
            #     r = 1   # the step that avoid detection

            # rewind the cycle idx to the beginning
            self.next_cycle_idx = 0

        if self.next_idx == self.NUM_TIME_SLOTS:
            # if it is the end, we stop it anyway
            d = True

        if d is True:
            # feed training data into Qnet
            if self.qNet is not None:
                self.qNet.train()

        return self.status, r, d, info

    def render(self):
        print "action = %d" % self.last_action

    def save_qnet_model(self):
        if self.qNet is not None:
            self.qNet.save_model()


class ClefQnet:
    def __init__(self, input_size, num_actions, config=None):
        self.learning_rate = 0.00001
        self.gamma = 0.99
        self.e = 0.2
        self.input_size = input_size
        self.num_actions = num_actions
        self.exp_name = "test_exp"
        self.batch_to_print_detail = 1000
        self.batch_idx = 0

        if config is not None:
            if "EXP_NAME" in config:
                self.exp_name = config["EXP_NAME"]

        N = self.input_size
        H = 2 * self.input_size
        T = self.num_actions
        self.inputs = tf.placeholder(shape=[None, N], dtype=tf.float32)
        self.W1 = tf.Variable(
            tf.truncated_normal([N, H], stddev=0.1), name='W1')
        self.b1 = tf.Variable(
            tf.truncated_normal([H], stddev=0.1), name='b1')
        z1 = tf.matmul(self.inputs, self.W1) + self.b1
        y1 = tf.nn.relu(z1)  # TODO: choose the right non-linear func

        self.W2 = tf.Variable(
            tf.truncated_normal([H, H], stddev=0.1), name='W2')
        self.b2 = tf.Variable(
            tf.truncated_normal([H], stddev=0.1), name='b2')
        z2 = tf.matmul(y1, self.W2) + self.b2
        y2 = tf.nn.relu(z2)  # TODO: choose the right non-linear func

        self.W3 = tf.Variable(
            tf.truncated_normal([H, H], stddev=0.1), name='W3')
        self.b3 = tf.Variable(
            tf.truncated_normal([H], stddev=0.1), name='b3')
        z3 = tf.matmul(y2, self.W3) + self.b3
        y3 = tf.nn.relu(z3)  # TODO: choose the right non-linear func

        self.W4 = tf.Variable(
            tf.truncated_normal([H, T], stddev=0.1), name='W4')
        self.b4 = tf.Variable(
            tf.truncated_normal([T], stddev=0.1), name='b4')
        z4 = tf.matmul(y3, self.W4) + self.b4
        self.prob = tf.nn.softmax(z4)

        # one hot input_y
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, T, dtype=tf.float32)
        self.advantages = tf.placeholder(
            shape=[None], dtype=tf.float32, name="reward_signal")
        self.loglik = tf.log(tf.reduce_sum(
            tf.multiply(self.prob, self.actions_onehot), reduction_indices=1))
        self.loss = -tf.reduce_mean(self.loglik * self.advantages)

        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.trainer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=1e-8)
        self.updateModel = self.trainer.minimize(self.loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

        # init experience buffer
        self.buffer = list()

    def _generate_input(self, t, status):
        if t == 0:
            return [0.5] * self.input_size
        if t > 0 and t < self.input_size:
            input_val = [0.5] * self.input_size
            input_val[self.input_size - t:] = status[:t]
        else:
            input_val = status[t - self.input_size:t]
        return input_val

    def add_to_exp_buffer(self, t, status, action, clef_cycle_reward):
        """clef_cycle_reward is the oversent traffic volume current cycle
        """
        s = self._generate_input(t, status)
        # buffer : [input, action,
        #   {total reward from after this action}]
        self.buffer.append([s, action, clef_cycle_reward])

    def train(self):
        self.batch_idx += 1
        # preprocess reward
        running_reward = 0.
        for i in reversed(xrange(0, len(self.buffer))):
            running_reward = self.gamma * running_reward + self.buffer[i][2]
            self.buffer[i][2] = running_reward
        self.buffer = np.array(self.buffer)
        # if len(self.buffer[:, 2]) > 1:
        #     self.buffer[:, 2] -= np.mean(self.buffer[:, 2])
        # std = np.std(self.buffer[:, 2])
        # if std > 0:
        #     self.buffer[:, 2] /= std
        inputs = np.vstack(self.buffer[:, 0])
        actions = self.buffer[:, 1]
        rewards = self.buffer[:, 2]

        # print "normalized actions = " + str(actions)
        # print "normalized reward = " + str(rewards)

        # update model
        self.sess.run(
            self.updateModel,
            feed_dict={
                self.inputs: inputs,
                self.actions: actions,
                self.advantages: rewards})

        # reset buffer
        self.buffer = list()

    def nextT_idx(self, t, status):
        inputs = self._generate_input(t, status)
        x = np.reshape(inputs, [1, len(inputs)])
        probs = self.sess.run(self.prob, feed_dict={self.inputs: x})
        rand = np.random.uniform()
        prob_accu = 0.
        T_idx = len(probs[0]) - 1
        for i in range(len(probs[0])):
            prob_accu += probs[0][i]
            if rand < prob_accu:
                T_idx = i
                break
        if np.random.uniform() < self.e:
            # still have a chance to try other T
            T_idx = np.random.randint(self.num_actions)
        # print "t = " + str(t) + " | s = " + str(x)
        if self.batch_idx % self.batch_to_print_detail == 0:
            print "QNet select T idx = " + str(T_idx) +\
                " | prob = " + str(probs)
        return T_idx

    def save_model(self):
        # save the model
        model_dir = './' + self.exp_name + '_model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.saver.save(self.sess, model_dir + '/clef_' + self.exp_name)

    def print_model(self):
        print "Final Model:"
        print "W1 = " + str(self.sess.run(self.W1))
        print "b1 = " + str(self.sess.run(self.b1))
        print "W2 = " + str(self.sess.run(self.W2))
        print "b2 = " + str(self.sess.run(self.b2))
        print "W3 = " + str(self.sess.run(self.W3))
        print "b3 = " + str(self.sess.run(self.b3))
        print "W4 = " + str(self.sess.run(self.W4))
        print "b4 = " + str(self.sess.run(self.b4))


if __name__ == '__main__':
    # for testing ClefEnv only
    import sys
    import json
    config = None   # using default setting
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        with open(sys.argv[2]) as config_file:
            config = json.load(config_file)
    env = ClefEnv(config=config)

    d = dict()
    num_T = len(env.T_LIST)
    for i in range(num_T):
        d[env.T_LIST[i]] = 0

    for _ in range(10000):
        T = env.T_LIST[env._pick_T_randomly()]
        d[T] += 1
    print "T frequency = " + str(d)

    # test ClefQnet
    qNet = ClefQnet(5, 4, config=None)
    status = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    qNet.add_to_exp_buffer(0, status, 0, -1)
    qNet.add_to_exp_buffer(4, status, 1, -1)
    qNet.add_to_exp_buffer(4, status, 1, -1)
    qNet.add_to_exp_buffer(4, status, 1, -1)
    qNet.add_to_exp_buffer(4, status, 1, -1)
    qNet.add_to_exp_buffer(4, status, 1, -1)

    qNet.train()
    print qNet.nextT_idx(8, status)
    qNet.print_model()
    qNet.save_model()
