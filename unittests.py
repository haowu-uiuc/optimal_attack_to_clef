import unittest
from brute_force import TrafficIterator
from utils import ValidChecker


class TestTrafficIterator(unittest.TestCase):

    def _traffic_to_int(self, traffic):
        x = 0
        for i in range(1, len(traffic)):
            x *= 2
            x += traffic[i]
        return x

    def test_iterating_all(self):
        period = 8
        it = TrafficIterator(period, checker=None)
        traffic = it.next()
        correct_code = 2 ** (period - 1) - 1
        while traffic is not None:
            code = self._traffic_to_int(traffic)
            self.assertEqual(code, correct_code)
            correct_code -= 1
            traffic = it.next()

    def test_iterating_non_detectable(self):
        period = 5
        checker = ValidChecker([1], 3)
        it = TrafficIterator(period, checker=checker)
        traffic = it.next()
        count = 0
        while traffic is not None:
            count += 1
            traffic = it.next()
        self.assertEqual(count, 11)

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == '__main__':
    unittest.main()
