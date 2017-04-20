import sys
import json
import numpy as np

T_LIST = range(1, 101, 5)
T_LIST = range(1, 11)
NUM_LEVELS = 4


def is_detectable(traffic, T, t):
    for start in range(
            t - T * NUM_LEVELS + 1,
            t - T * (NUM_LEVELS - 1) + 1):
        probs = [0] * NUM_LEVELS
        for i in range(start, start + NUM_LEVELS * T):
            if i < 0:
                continue
            if i >= len(traffic):
                break
            if traffic[i] == 1:
                probs[(i - start) / T] = 1
            if (i - start + 1) % T == 0 and probs[(i - start) / T] == 0:
                # (start, T) cannot detect this traffic
                break
        if sum(probs) == NUM_LEVELS:
            return True
    return False


def render_traffic(traffic):
    for i in range(0, len(traffic)):
        if traffic[i] == 1:
            print "%d ->\t [*]" % i
        else:
            print "%d ->\t [ ]" % i
    print "rate = " + str(float(sum(traffic)) / len(traffic))


if __name__ == '__main__':
    config = None   # using default setting
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        with open(sys.argv[2]) as config_file:
            config = json.load(config_file)

    if config is not None:
        if "T_LIST" in config:
            T_LIST = config["T_LIST"]
        if "NUM_LEVELS" in config:
            NUM_LEVELS = config["NUM_LEVELS"]

    max_T = max(T_LIST)
    max_rate = 0.
    max_traffic = []
    rate_sum = 0.
    tmp_traffic_sum = 0
    for k in range(1):
        traffic = [0] * (max_T * NUM_LEVELS * 10)

        for i in range(len(traffic)):
            if np.random.uniform() > 0.0:
                traffic[i] = 1
                tmp_traffic_sum += 1
                # try all T and sliding window to see whether it is detected
                for T in T_LIST:
                    if is_detectable(traffic, T, i):
                        traffic[i] = 0
                        tmp_traffic_sum -= 1
                        break

        rate = float(tmp_traffic_sum) / len(traffic)
        if max_rate < rate:
            max_traffic = traffic
            max_rate = rate
        tmp_traffic_sum = 0
        rate_sum += rate
        print str(k) + ":\tave rate = " + str(rate_sum / (k + 1))
        # render_traffic(traffic)
    render_traffic(max_traffic)
    print "max rate = " + str(max_rate)
