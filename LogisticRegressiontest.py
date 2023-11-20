import unittest
from LR import LogisticRegression
import numpy as np

class TestLogisticRegression(unittest.TestCase):
    def test_init(self):
        lr = LogisticRegression(0.01, 50, 100, 1e-3)
        self.assertEqual(lr.eta, 0.01, "incorrect eta")

    def test_import_weight(self):
        lr = LogisticRegression()
        lr.import_weights('weights.txt')

        w = [-0.08361324, -1.69299114, 1.83990052]
        w = np.array(w)

        for i in range(len(w)):
            self.assertEqual(w[i], lr.w__[i])