import sys
import json
import numpy as np
from utils import ValidChecker
from utils import render_traffic

T_LIST = range(1, 6)
NUM_LEVELS = 4
MAX_PERIOD = 20     # maximum period size to brute foce to


class TrafficIterator:
    def __init__(self, period, checker=None):
        self.period = period
        self.stack = list()
        self.stack.append(1)    # the first slot of a period must be '1'
        self.is_end = False
        self.checker = checker
        self._explore_forward()

    def _explore_forward(self):
        while len(self.stack) < self.period:
            self.stack.append(1)
            if self.checker is not None and\
                    self.checker.is_detectable_at_t(
                        self.stack, len(self.stack) - 1):
                # skip detectable traffic at very beginning
                self.stack.pop()
                self.stack.append(0)

    def _goto_next(self):
        while len(self.stack) > 1 and self.stack[-1] == 0:
            self.stack.pop()     # pop trailing zeros in the current branch
        if len(self.stack) == 1:
            # we have traversed all patterns
            self.is_end = True
            return

        self.stack.pop()    # pop the last '1'
        self.stack.append(0)
        self._explore_forward()

    def next(self):
        if self.is_end is not True:
            next_traffic = list(self.stack)
            self._goto_next()
            return next_traffic
        return None


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

    p = MAX_PERIOD
    T_max = max(T_LIST)
    checker = ValidChecker(T_LIST, NUM_LEVELS)
    max_rate = 0.
    max_pattern = []
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

        it = TrafficIterator(p, checker=checker)
        pattern = it.next()
        while pattern is not None:
            num_periods = (NUM_LEVELS * T_max / p + 1) + 1
            traffic = pattern * num_periods
            # print "traffic: " + str(traffic)
            if not checker.is_detectable(traffic, period=p):
                rate = float(sum(pattern)) / p
                if rate > max_rate:
                    max_rate = rate
                    max_pattern = pattern
            pattern = it.next()

    print "max rate = %f" % max_rate
    render_traffic(max_pattern)
