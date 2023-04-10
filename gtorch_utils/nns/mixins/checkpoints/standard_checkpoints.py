# -*- coding: utf-8 -*-
""" gtorch_utils/nns/mixins/checkpoints/standard_checkpoints.py """

from gtorch_utils.nns.mixins.checkpoints.base import CheckPointBaseMixin


__all__ = ['CheckPointMixin']


class CheckPointMixin(CheckPointBaseMixin):
    """
    Provides standard methods to save and load checkpoints for inference or resume training

    Usage:
        class SomeClass(CheckPointMixin):
            ...
    """
