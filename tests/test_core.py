import unittest
from termsheet_analyser.step02_extraction import extract_structural_tags, _norm_text, _norm_value

class CoreTests(unittest.TestCase):
    def test_autocall_and_frequency(self):
        r=extract_structural_tags("Memory quarterly step-down autocall worst-of note")
        self.assertTrue(r["is_autocall"])
        self.assertTrue(r["is_memory"])
        self.assertTrue(r["is_step_down"])
        self.assertTrue(r["is_worst_of"])
        self.assertEqual(r["frequency"],"Quarterly")
    def test_empty_text(self):
        r=extract_structural_tags("")
        self.assertFalse(r["is_autocall"])
        self.assertEqual(r["frequency"],"Unknown")
    def test_normalise(self):
        self.assertEqual(_norm_text("  Coupon   Barrier "),"coupon barrier")
        self.assertEqual(_norm_value(" 1,000 % "),"1000%")

if __name__=="__main__":unittest.main()
