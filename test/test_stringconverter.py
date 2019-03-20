"""
Tests of Stringconverter.
run with
`pytest ./test_stringconverter.py`
"""

import unittest
from dias.utils import str2bytes, bytes2str


class TestStringConverter(unittest.TestCase):

    def test_str2bytes(self):

        assert str2bytes("0 GB") == 0
        assert str2bytes("0 B") == 0
        assert str2bytes("1 B") == 1
        assert str2bytes("1024 B") == 1024
        assert str2bytes("3 kB") == 3000
        assert str2bytes("27 MB") == 27000000
        assert str2bytes("1300 GB") == 1300000000000

        # Missing whitespace
        with self.assertRaises(ValueError):
            str2bytes("0B")

        with self.assertRaises(ValueError):
            str2bytes("B")
        with self.assertRaises(ValueError):
            str2bytes("0")
        with self.assertRaises(ValueError):
            str2bytes("")
        with self.assertRaises(ValueError):
            str2bytes(1)

    def test_bytes2str(self):
        assert bytes2str(0) == '0.0 B'
        assert bytes2str(1) == '1.0 B'
        assert bytes2str(1100) == '1.1 kB'
        assert bytes2str(9919999) == '9.9 MB'
        assert bytes2str(9999999) == '10.0 MB'
