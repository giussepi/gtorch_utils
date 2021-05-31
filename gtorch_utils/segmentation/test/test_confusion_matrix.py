# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/test/test_confusion_matrix """

import unittest

import torch

from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


class Test_ConfusionMatrixMGR(unittest.TestCase):

    def setUp(self):
        self.gt = torch.Tensor([[[[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]]]])
        self.preds_binary = torch.Tensor([[[[1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]])
        self.preds = torch.Tensor([[[[.8, .1, .3, .4], [.4, .3, .7, .3], [.1, .5, .8, .2], [.2, .3, .4, .1]]]])

    def test_complement_binary(self):
        self.assertTrue(torch.equal(
            ConfusionMatrixMGR.complement(self.preds_binary) + self.preds_binary,
            torch.ones(self.preds_binary.size())
        ))

    def test_complement(self):
        self.assertTrue(torch.equal(
            ConfusionMatrixMGR.complement(self.preds) + self.preds,
            torch.ones(self.preds.size())
        ))

    def test_true_positives_binary(self):
        self.assertEqual(ConfusionMatrixMGR(self.preds_binary, self.gt).true_positives, 3)

    def test_true_positives(self):
        self.assertEqual(ConfusionMatrixMGR(self.preds, self.gt).true_positives, 3)

    def test_false_positives_binary(self):
        self.assertEqual(ConfusionMatrixMGR(self.preds_binary, self.gt).false_positives, 2)

    def test_false_positives(self):
        self.assertEqual(round(ConfusionMatrixMGR(self.preds, self.gt).false_positives.item(), 1), 2.9)

    def test_true_negatives_binary(self):
        self.assertEqual(ConfusionMatrixMGR(self.preds_binary, self.gt).true_negatives, 8)

    def test_true_negatives(self):
        self.assertEqual(round(ConfusionMatrixMGR(self.preds, self.gt).true_negatives.item(), 1), 7.1)

    def test_false_negatives_binary(self):
        self.assertEqual(ConfusionMatrixMGR(self.preds_binary, self.gt).false_negatives, 3)

    def test_false_negatives(self):
        self.assertEqual(round(ConfusionMatrixMGR(self.preds, self.gt).false_negatives.item(), 1), 3.)

    def test_all_values_binary(self):
        tp, fp, tn, fn = ConfusionMatrixMGR(self.preds_binary, self.gt)()
        self.assertEqual((tp, fp, tn, fn), (3, 2, 8, 3))

    def test_all_values(self):
        tp, fp, tn, fn = ConfusionMatrixMGR(self.preds, self.gt)()
        self.assertEqual(
            (tp, round(fp.item(), 1), round(tn.item(), 1),
             round(fn.item(), 1)), (3, 2.9, 7.1, 3)
        )

    def test_summation_equivalence_TP_FN(self):
        binary = ConfusionMatrixMGR(self.preds_binary, self.gt)
        non_binary = ConfusionMatrixMGR(self.preds, self.gt)

        self.assertTrue(torch.equal(
            binary.true_positives + binary.false_negatives,
            non_binary.true_positives + non_binary.false_negatives
        ))

    def test_summation_equivalence_FP_TN(self):
        binary = ConfusionMatrixMGR(self.preds_binary, self.gt)
        non_binary = ConfusionMatrixMGR(self.preds, self.gt)

        self.assertTrue(torch.equal(
            binary.false_positives + binary.true_negatives,
            non_binary.false_positives + non_binary.true_negatives
        ))


if __name__ == '__main__':
    unittest.main()
