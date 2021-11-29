# -*- coding: utf-8 -*-
""" gtorch_utils/nns/managers/callbacks/checkpoint """


class Checkpoint:
    """
    Checkpoint interval evaluator

    Usage:
        checkpoint = Checkpoint(5)
        checkpoint(iteration)
    """

    def __init__(self, interval=5):
        """ Initialized the instance """
        assert isinstance(interval, int)
        assert interval > 0

        self.interval = interval

    def __call__(self, iteration):
        """ Functor call """
        return self.evaluate(iteration)

    def evaluate(self, iteration):
        """
        Returns True if the current iteration must be saved, otherwise returns False

        Args:
            iteration (int) : zero-based iterarion

        Returns:
            bool
        """
        return (iteration + 1) % self.interval == 0
