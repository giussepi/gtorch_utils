# -*- coding: utf-8 -*-
""" gtorch_utils/nns/managers/callbacks/memoryprint """

import subprocess

from logzero import logger


__all__ = ['MemoryPrint']


class MemoryPrint:
    """
    Class to collect and print GPU and CPU memory stats

    Usage:
        memory_printer = MemoryPrint(10)  # initialization
        memory_printer(0, initial_stats=True)  # to be called before loading your model
        memory_printer(epoch)  # to be called during training
    """

    def __init__(self, epoch_interval: int, shell_command: str = '', gpu: bool = True):
        """ Initializes the object instance """
        assert isinstance(gpu, bool), type(gpu)
        assert isinstance(epoch_interval, int), type(epoch_interval)
        assert isinstance(shell_command, str), type(shell_command)
        assert epoch_interval >= 0, epoch_interval

        self.gpu = gpu
        self.epoch_interval = epoch_interval

        if shell_command:
            self.shell_command = shell_command
        else:
            self.shell_command = 'nvidia-smi' if self.gpu else 'free -m'

    def __call__(self, current_epoch: int, /, *, initial_stats: bool = False):
        self.print_memory_stats(current_epoch, initial_stats=initial_stats)

    def print_memory_stats(self, current_epoch: int, /, *, initial_stats: bool = False):
        """
        Prints memory start depending on current_epoch and self.epoch_interval

        Kwargs:
            current_epoch  <int>: Zero-based current epoch
            initial_stats <bool>: If set to True, a message saying that it's the first
                                  memory start before loading the model is shown.
        """
        assert isinstance(current_epoch, int), type(current_epoch)
        assert current_epoch >= 0, current_epoch
        assert isinstance(initial_stats, bool), type(initial_stats)

        if self.epoch_interval == 0:
            return

        if initial_stats:
            logger.info('Memory print before loading the model:')

        if current_epoch % self.epoch_interval == 0:
            try:
                with subprocess.Popen(
                        self.shell_command.split(), cwd='.', stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT
                ) as process:
                    output, _ = process.communicate()
                    logger.info(output.decode('utf-8'))
            except (OSError, subprocess.CalledProcessError) as exception:
                logger.warning(exception)
