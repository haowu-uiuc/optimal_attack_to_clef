import sys
import json
from utils import render_traffic
from utils import TrafficIterator
from clef_env import ClefEnv
import operator
import os.path

T_LIST = range(1, 3)
NUM_LEVELS = 4
MAX_PERIOD = 5     # maximum period size to brute foce to
NUM_TEST_EPISODES = 1000
THRESHOLD_RATE = 0.333333333333
EXP_NAME = "test_bf_exp"
OUTPUT_DIR = "."

if __name__ == '__main__':
    config = None   # using default setting
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        with open(sys.argv[2]) as config_file:
            config = json.load(config_file)
        OUTPUT_DIR = '/'.join(
            os.path.abspath(config_file.name).split('/')[:-1])

    if config is not None:
        if "T_LIST" in config:
            T_LIST = config["T_LIST"]
        if "NUM_LEVELS" in config:
            NUM_LEVELS = config["NUM_LEVELS"]
        if "THRESHOLD_RATE" in config:
            THRESHOLD_RATE = config["THRESHOLD_RATE"]
        if "MAX_PERIOD" in config:
            MAX_PERIOD = config["MAX_PERIOD"]
        if "EXP_NAME" in config:
            EXP_NAME = config["EXP_NAME"]
        if "NUM_TEST_EPISODES" in config:
            NUM_TEST_EPISODES = config["NUM_TEST_EPISODES"]

    p = MAX_PERIOD
    T_max = max(T_LIST)
    damages = dict()    # rate :-> (max damage at this rate, life_time)
    max_life_times = dict()
    patterns = dict()
    detector = ClefEnv(config=config)
    # traverse the traffic pattern from MAX_PERIOD to 2
    # skip period p if there is a p' > p and p' % p == 0,
    # to avoid considering duplicated pattern
    for p in range(MAX_PERIOD, 2, -1):
        # deduplication
        is_duplicate = False
        for checked_p in range(MAX_PERIOD, p, -1):
            if checked_p % p == 0:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        it = TrafficIterator(p, checker=None)
        pattern = it.next()
        while pattern is not None:
            print pattern
            ave_volume = 0
            ave_life_time = 0
            for i in range(NUM_TEST_EPISODES):
                # calculate the average traffic volume by repeated experiments
                detector.reset()
                d = False
                t = 0
                while d is not True:
                    action = pattern[t % p]
                    ave_volume += action
                    t += 1
                    _, _, d, _ = detector.step(action)
                ave_life_time += t
            ave_volume /= float(NUM_TEST_EPISODES)
            ave_life_time /= float(NUM_TEST_EPISODES)
            ave_damage = ave_volume - ave_life_time * THRESHOLD_RATE
            print ave_life_time
            print ave_volume
            pattern_rate = sum(pattern) / float(p)
            if pattern_rate not in damages:
                damages[pattern_rate] = (ave_damage, ave_life_time)
                patterns[pattern_rate] = pattern
                max_life_times[pattern_rate] = ave_life_time
            else:
                if damages[pattern_rate][0] < ave_damage:
                    damages[pattern_rate] = (ave_damage, ave_life_time)
                    patterns[pattern_rate] = pattern
                if max_life_times[pattern_rate] < ave_life_time:
                    max_life_times[pattern_rate] = ave_life_time
            pattern = it.next()

    sorted_damages = sorted(damages.items(), key=operator.itemgetter(0))
    sorted_life_times = sorted(
        max_life_times.items(), key=operator.itemgetter(0))
    sorted_patterns = sorted(patterns.items(), key=operator.itemgetter(0))
    print sorted_damages
    with open(OUTPUT_DIR + "/bf_damage_" + EXP_NAME + ".txt", 'w') as f:
        for rate, value in sorted_damages:
            # rate, damage, life_time
            f.write("%f\t%f\t%f\n" % (rate, value[0], value[1]))

    with open(OUTPUT_DIR + "/bf_max_life_time_" + EXP_NAME + ".txt", 'w') as f:
        for rate, life_time in sorted_life_times:
            f.write(str(rate) + "\t" + str(life_time) + "\n")

    with open(OUTPUT_DIR + "/bf_patterns_" + EXP_NAME + ".txt", 'w') as f:
        for rate, pattern in sorted_patterns:
            f.write(str(rate) + "\t" + pattern + "\n")
