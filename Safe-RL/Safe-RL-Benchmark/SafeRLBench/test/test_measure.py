from SafeRLBench.measure import BestPerformance, SafetyMeasure

from mock import Mock
from unittest2 import TestCase


def _mock_run(val):
    run = Mock()
    monitor = Mock()
    monitor.rewards = range(val, val + 4)
    run.get_alg_monitor.return_value = monitor

    print(monitor.rewards)
    print(run.get_alg_monitor())
    print(monitor)

    return run


class TestMeasure(TestCase):
    """Test Measure classes."""

    def test_best_performance(self):
        """Test: MEASURE: BestPerformance."""
        run1 = _mock_run(0)
        run2 = _mock_run(1)

        measure = BestPerformance()
        self.assertIsNone(measure.result)

        measure([run1, run2])
        result = measure.result

        self.assertEquals(result[0][0], run2)
        self.assertEquals(result[1][0], run1)

        self.assertEquals(result[0][1], 4)
        self.assertEquals(result[1][1], 3)

        best_result = measure.best_result

        self.assertEquals(best_result[0], run2)
        self.assertEquals(best_result[1], 4)

    def test_safety_measure(self):
        """Test: MEASURE: SafetyMeasure."""
        measure = SafetyMeasure(0)
        self.assertIsNone(measure.result)

        run1 = _mock_run(-2)
        run2 = _mock_run(0)

        measure([run1, run2])

        result = measure.result

        self.assertEquals(result[0][0], run1)
        self.assertEquals(result[0][1], 2)
        self.assertEquals(result[0][2], 3)

        self.assertEquals(result[1][0], run2)
        self.assertEquals(result[1][1], 0)
        self.assertEquals(result[1][2], 0)
