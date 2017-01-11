from __future__ import division

import nose.tools

from rfcn import utils


def test_get_bbox_overlap():
    # (x1, y1, x2, y2)
    bbox1 = (0, 0, 10, 10)
    # fully overlap
    bbox2 = (0, 0, 10, 10)
    expected = 1
    actual = utils.get_bbox_overlap(bbox1, bbox2)
    nose.tools.assert_equal(actual, expected)
    # no overlap
    bbox2 = (11, 12, 21, 22)
    expected = 0
    actual = utils.get_bbox_overlap(bbox1, bbox2)
    nose.tools.assert_equal(actual, expected)
    # inside overlap
    bbox2 = (3, 5, 5, 7)
    expected = 2*2 / (10*10 + 2*2 - 2*2)
    actual = utils.get_bbox_overlap(bbox1, bbox2)
    nose.tools.assert_equal(actual, expected)
    # partly overlap 1
    bbox2 = (3, 5, 4, 15)
    expected = 1*5 / (10*10 + 1*10 - 1*5)
    actual = utils.get_bbox_overlap(bbox1, bbox2)
    nose.tools.assert_equal(actual, expected)
    # partly overlap 2
    bbox2 = (3, 5, 13, 6)
    expected = 1*7 / (10*10 + 10*1 - 1*7)
    actual = utils.get_bbox_overlap(bbox1, bbox2)
    nose.tools.assert_equal(actual, expected)
    # partly overlap 3
    bbox2 = (3, 5, 13, 15)
    expected = 7*5 / (10*10 + 10*10 - 7*5)
    actual = utils.get_bbox_overlap(bbox1, bbox2)
    nose.tools.assert_equal(actual, expected)
