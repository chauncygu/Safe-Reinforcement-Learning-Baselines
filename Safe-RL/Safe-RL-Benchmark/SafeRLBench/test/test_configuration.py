from SafeRLBench import SRBConfig

from unittest2 import TestCase

import sys
import os

import logging

logger = logging.getLogger(__name__)


class TestSRBConfig(TestCase):
    """Test SRBConfig class."""

    def test_logger_stream_handler(self):
        """Test: CONFIG: stream handler."""
        config = SRBConfig(logger)

        self.assertIsNone(config.logger_stream_handler)

        # check if stream handler gets added
        config.logger_add_stream_handler()
        self.assertIsNotNone(config.logger_stream_handler)

        handler1 = config.logger_stream_handler
        handler2 = logging.StreamHandler(sys.stdout)

        # check if handler changes on assignment
        config.logger_stream_handler = handler2
        self.assertNotEqual(handler1, config.logger_stream_handler)

    def test_logger_file_handler(self):
        """Test: CONFIG: file handler."""
        config = SRBConfig(logger)

        self.assertIsNone(config.logger_file_handler)

        # check if file handler gets added
        config.logger_add_file_handler('logs.log')
        self.assertIsNotNone(config.logger_file_handler)

        handler1 = config.logger_file_handler
        handler2 = logging.FileHandler('logs2.log')

        # check if handler changes on assignment
        config.logger_file_handler = handler2
        self.assertNotEqual(handler1, config.logger_file_handler)

        self.assertTrue(os.path.isfile('logs.log'))
        self.assertTrue(os.path.isfile('logs2.log'))

        config.logger_file_handler = None

    def test_logger_format(self):
        """Test: CONFIG: logger format."""
        config = SRBConfig(logger)

        config.logger_add_stream_handler()
        config.logger_add_file_handler('logs.log')

        fmt = '%(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)

        config.logger_format = fmt

        tst_record = {
            'name': 'test_logger',
            'level': logging.DEBUG,
            'pathname': os.path.realpath(__file__),
            'lineno': 42,
            'msg': 'test_msg',
            'args': None,
            'exc_info': None,
            'func': 'test_logger_format'
        }
        rec = logging.makeLogRecord(tst_record)
        self.assertEqual(formatter.format(rec),
                         config.logger_stream_handler.format(rec))

    def test_monitor_verbosity(self):
        """Test: CONFIG: monitor verbosity."""
        config = SRBConfig(logger)

        config.monitor_set_verbosity(42)
        self.assertEqual(config.monitor_verbosity, 42)

        with self.assertRaises(ValueError):
            config.monitor_set_verbosity(-1)

    def test_jobs(self):
        """Test: CONFIG: jobs set."""
        config = SRBConfig(logger)

        config.jobs_set(42)
        self.assertEqual(config.n_jobs, 42)

        with self.assertRaises(ValueError):
            config.jobs_set(-1)

    @classmethod
    def tearDownClass(cls):
        """Clean up created file."""
        if os.path.isfile('logs.log'):
            os.remove('logs.log')
        if os.path.isfile('logs2.log'):
            os.remove('logs2.log')
