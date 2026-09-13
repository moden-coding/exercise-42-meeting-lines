#!/usr/bin/env python3
"""Tests for the Meeting Lines assignment."""

import unittest
from unittest.mock import patch

import numpy as np

from src.meeting_lines import meeting_lines


class TestMeetingLines(unittest.TestCase):
    """meeting_lines(a1, b1, a2, b2) -> (x, y) where y=a1*x+b1=a2*x+b2."""

    def test_first(self):
        a1 = 1
        b1 = 4
        a2 = 3
        b2 = 2
        x, y = meeting_lines(a1, b1, a2, b2)

        self.assertAlmostEqual(
            y,
            a1 * x + b1,
            msg="Meeting point (%r, %r) is not on line a1=%i, b1=%i"
            % (x, y, a1, b1),
        )
        self.assertAlmostEqual(
            y,
            a2 * x + b2,
            msg="Meeting point (%r, %r) is not on line a2=%i, b2=%i"
            % (x, y, a2, b2),
        )

    def test_calls(self):
        with patch("numpy.linalg.solve", wraps=np.linalg.solve) as psolve:
            a1 = 1
            b1 = 4
            a2 = 3
            b2 = 2
            meeting_lines(a1, b1, a2, b2)
            self.assertTrue(
                psolve.called,
                msg="meeting_lines should solve the linear system with "
                "np.linalg.solve.",
            )

    def test_underdetermined(self):
        a1 = 1
        b1 = 4
        p = (a1, b1, a1, b1)
        system = "(a1=%i, b1=%i, a2=%i, b2=%i)" % p
        with self.assertRaises(
            np.linalg.LinAlgError,
            msg="Under determined system %s should raise an exception!"
            % system,
        ):
            meeting_lines(*p)

    def test_inconsistent(self):
        a1 = 1
        b1 = 4
        p = (a1, b1, a1, b1)
        system = "(a1=%i, b1=%i, a2=%i, b2=%i)" % p
        with self.assertRaises(
            np.linalg.LinAlgError,
            msg="Inconsistent system %s should raise an exception!" % system,
        ):
            meeting_lines(*p)


if __name__ == "__main__":
    unittest.main()
