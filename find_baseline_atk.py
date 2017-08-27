import sys
import json
import numpy as np
from utils import ValidChecker
from utils import render_traffic

T_LIST = range(1, 101, 5)
T_LIST = range(1, 11)
NUM_LEVELS = 4
NUM_DET_CYCLE = 10

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

    checker = ValidChecker(T_LIST, NUM_LEVELS)

    max_T = max(T_LIST)
    max_rate = 0.
    max_traffic = []
    rate_sum = 0.
    tmp_traffic_sum = 0
    for k in range(1):
        traffic = [0] * (max_T * NUM_LEVELS * NUM_DET_CYCLE)

        for i in range(len(traffic)):
            if np.random.uniform() > 0.0:
                traffic[i] = 1
                tmp_traffic_sum += 1
                # try all T and sliding window to see whether it is detected
                if checker.is_detectable_at_t(traffic, i):
                    traffic[i] = 0
                    tmp_traffic_sum -= 1

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
