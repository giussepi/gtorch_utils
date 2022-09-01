# -*- coding: utf-8 -*-
""" gtorch_utils/nns/utils/reproducibility """

import os
import random

import numpy as np
import torch


__all__ = ['Reproducibility']


class Reproducibility:
    """
    Holds methods to allow reproducibility in Pytorch

    See https://pytorch.org/docs/stable/notes/randomness.html

    Usage:
        # before your main definition
        reproducibility = Reproducibility()

        # when creating a DataLoader instance
        DataLoader(
           train_dataset,
           batch_size=batch_size,
           num_workers=num_workers,
           worker_init_fn=Reproducibility.seed_worker,
           generator=reproducibility.get_generator(),
       )
    """

    def __init__(self, seed_value: int = 20, /, *, cuda: bool = False, disable_cuda_benchmark: bool = True,
                 deterministic_algorithms: bool = True, cublas_env_vars: bool = False,
                 cuda_conv_determinism: bool = True):
        """
        Initializes the object instance and sets the configurations to have a deterministic behaviour

        Kwargs:
            seed_value <int>: Seed value to be used . Default 20
            cuda <bool>: Whether to apply deterministic behaviour to CUDA or not. Default False
            disable_cuda_benchmark <bool>: Whether or disable CUDA benchmark to let cuDNN to deterministically
                         select an algorithm, possibly at the cost of reduced performance. While disabling
                         CUDA convolution benchmarking ensures that CUDA selects the same algorithm
                         each time an application is run, that algorithm itself may be nondeterministic,
                         unless either deterministic_algorithms=True or cuda_conv_determinism=True is set.
                         Default True
            deterministic_algorithms <bool>: Whether or not use deterministic algorithms. However, not all
                         layers have a deterministic version. Review
                         https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                         If you are using CUDA tensors, and your CUDA version is 10.2 or greater,
                         you should set the environment variable CUBLAS_WORKSPACE_CONFIG according to CUDA
                         documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
                         This can be done by setting cublas_env_vars = True
                         Default True

            cublas_env_vars <bool>: Whether of not set CUBLAS_WORKSPACE_CONFIG to allow reproducibility
                         behaviour. It requires deterministic_algorithms = True.
                         Default False
            cuda_conv_determinism <bool> Whether or not use deterministic convolutions. Default True
        """
        assert isinstance(seed_value, int), type(seed_value)
        assert seed_value >= 0, seed_value
        assert isinstance(cuda, bool), type(cuda)
        assert isinstance(disable_cuda_benchmark, bool), type(disable_cuda_benchmark)
        assert isinstance(deterministic_algorithms, bool), type(deterministic_algorithms)
        assert isinstance(cublas_env_vars, bool), type(cublas_env_vars)
        assert isinstance(cuda_conv_determinism, bool), type(cuda_conv_determinism)

        self.seed_value = seed_value
        self.cuda = cuda
        self.disable_cuda_benchmark = disable_cuda_benchmark
        self.deterministic_algorithms = deterministic_algorithms
        self.cublas_env_vars = cublas_env_vars
        self.cuda_conv_determinism = cuda_conv_determinism

        self._manual_seeding()

    def _manual_seeding(self):
        """
        Sets the configuration to have a deterministic behaviour
        """
        random.seed(self.seed_value)  # cpu vars
        np.random.seed(self.seed_value)  # cpu vars
        torch.manual_seed(self.seed_value)  # cpu vars

        if self.cuda:
            # Sets the seed for generating random numbers on all GPUs.
            torch.cuda.manual_seed_all(self.seed_value)
            torch.backends.cudnn.benchmark = not self.disable_cuda_benchmark

            torch.use_deterministic_algorithms(self.deterministic_algorithms)

            if self.deterministic_algorithms and self.cublas_env_vars:
                # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # may limit overall performance
                # will increase library footprint in GPU memory by approximately 24MiB
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

            torch.backends.cudnn.deterministic = self.cuda_conv_determinism

    @staticmethod
    def seed_worker(worker_id: int):
        """
        Worker init static method to be used with DataLoader
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_generator(self) -> torch._C.Generator:
        """
        Returns a Generator instance to be used with the DataLoader
        """
        g = torch.Generator()
        g.manual_seed(self.seed_value)

        return g
