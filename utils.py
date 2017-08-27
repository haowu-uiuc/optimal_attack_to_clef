
class ValidChecker:
    def __init__(self, Ts, num_levels):
        self.Ts = Ts
        self.num_levels = num_levels

    def is_detectable(self, traffic, period=None):
        start = 0
        if period is not None:
            start = len(traffic) - period
        for t in range(start, len(traffic)):
            for T in self.Ts:
                if self._detect_at_t_by_T(traffic, T, t):
                    return True
        return False

    def is_detectable_at_t(self, traffic, t):
        """check whether the traffic at t can cause a detection
        in the greedy algorithm"""
        for T in self.Ts:
            for tx in range(t, t + T):
                if self._detect_at_t_by_T(traffic, T, tx):
                    return True
        return False

    def _detect_at_t_by_T(self, traffic, T, t):
        """detect traffic with detection window ending at t"""
        start = t - T * self.num_levels + 1
        probs = [0] * self.num_levels
        for i in range(start, t + 1):
            if i < 0:
                continue
            if i >= len(traffic):
                break
            if traffic[i] == 1:
                probs[(i - start) / T] = 1
            if (i - start + 1) % T == 0 and probs[(i - start) / T] == 0:
                # (start, T) cannot detect this traffic
                break
        if sum(probs) == self.num_levels:
            return True
        return False


class TrafficIterator:
    """Iterator to traverse all possible traffic pattern given a traffic period
    If we specify the checker, the iterator will skip those detectable traffic
    patterns.
    """
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


def render_traffic(traffic):
    for i in range(0, len(traffic)):
        if traffic[i] == 1:
            print "%d ->\t [*]" % i
        else:
            print "%d ->\t [ ]" % i
    print "rate = " + str(float(sum(traffic)) / len(traffic))
