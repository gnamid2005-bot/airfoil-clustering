import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from continue_gabriel_k6_k9 import k6_screen_present


class ExactScreenMatchTest(unittest.TestCase):
    def test_coordinator_name_does_not_match(self):
        self.assertFalse(k6_screen_present('  64275.gabriel_k6_k9_queue  (Detached)\n'))

    def test_exact_k6_name_matches(self):
        self.assertTrue(k6_screen_present('  12345.gabriel_k6  (Detached)\n'))

    def test_both_names_find_exact_k6(self):
        output='  64275.gabriel_k6_k9_queue  (Detached)\n  12345.gabriel_k6  (Detached)\n'
        self.assertTrue(k6_screen_present(output))


if __name__=='__main__':
    unittest.main()
