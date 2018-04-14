import math
import numpy as np
import pytest

import preprocess
import analysis


@pytest.mark.parametrize(
   'test_type, var1, var2, expect', [
  ('verify_max_returned', 'apple', 90, 100),
  ('verify_min_returned', 'orange', 2, 10)
])
def test_preprocess(test_type, var1, var2, expect):
  print("Logging test type for visibility: " + test_type)
  assert 5 == 5

def test_accuracy_with_complete_overlap():
    o_mask = np.zeros((25,25,1))
    o_mask[5:20,5:20,0] = 1

    i_mask = np.zeros((25,25,1))
    i_mask[10:15,10:15,0] = 1

    pred_i_mask = np.zeros((25,25,1))
    pred_i_mask[i_mask == 1] = 1

    score = analysis.accuracy(o_mask, i_mask, pred_i_mask)
    assert score == 1.0

def test_accuracy_with_no_overlap():
    o_mask = np.zeros((25,25,1))
    o_mask[5:20,5:20,0] = 1

    i_mask = np.zeros((25,25,1))
    i_mask[15:20,15:20,0] = 1

    pred_i_mask = np.ones((25,25,1))
    pred_i_mask[i_mask == 1] = 0

    score = analysis.accuracy(o_mask, i_mask, pred_i_mask)
    assert score == 0.0

def test_accuracy_with_partial_overlap():
    o_mask = np.zeros((25,25,1))
    o_mask[5:20,5:20,0] = 1

    i_mask = np.zeros((25,25,1))
    i_mask[10:15,10:15,0] = 1

    pred_i_mask = np.zeros((25,25,1))
    pred_i_mask[9:14,10:15,0] = 1

    score = analysis.accuracy(o_mask, i_mask, pred_i_mask)
    assert math.isclose(score, .955, rel_tol=1e-3, abs_tol=0.0)
