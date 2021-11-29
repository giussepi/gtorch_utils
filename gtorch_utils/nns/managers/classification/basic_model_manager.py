# -*- coding: utf-8 -*-
""" gtorch_utils/nns/managers/classification/basic_model_manager """

import os
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gutils.decorators import timing
from torch.utils.data import DataLoader

from gtorch_utils.constants import DB
from gtorch_utils.datasets.generic import BaseDataset
from gtorch_utils.nns.managers.callbacks import EarlyStopping, Checkpoint, PlotTensorBoard


class BasicModelMGR:
    """
    General model managers

    Usage:

        # Minimum configuration required
        BasicModelMGR(
            model=Perceptron(3000, 102),
            dataset=GenericDataset,
            dataset_kwargs=dict(dbhandler=DBhandler, normalizer=Normalizer.MAX_NORM, val_size=.1),
            epochs=200
        )()

        # Full options
        BasicModelMGR(
            cuda=True,
            model=Perceptron(3000, 102),
            sub_datasets=DB,
            dataset=GenericDataset,
            dataset_kwargs=dict(dbhandler=DBhandler, normalizer=Normalizer.MAX_NORM, val_size=.1),
            batch_size=6,
            shuffe=False,
            num_workers=12,
            optimizer=optim.SGD,
            optimizer_kwargs=dict(lr=1e-1, momentum=.9),
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            epochs=200,
            earlystopping_kwargs=dict(min_delta=1e-2, patience=8),
            checkpoints=True,
            checkpoint_interval=5,
            checkpoint_path=OrderedDict(directory_path='tmp', filename=''),
            saving_details=OrderedDict(directory_path='tmp', filename='best_model.pth'),
            tensorboard=True
        )()
    """

    model_extensions = ('.pth', '.pt')
    checkpoint_extensions = ('.pth.tar', 'pt.tar')
    checkpoint_pattern = 'chkpt_{}.pth.tar'

    def __init__(self, **kwargs):
        """
        kwargs:
            cuda (bool): whether or not use cuda
            model (nn.Module): model instance
            sub_datasets (class): class containing the subdatasets. See
                                  gtorch_utils.models.constants.DB class definition
            dataset (BaseDataset): Custom dataset class descendant of gtorch_utils.datasets.generic.BaseDataset. Also see https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
            dataset_kwargs (dict): keyword arguments for the dataset
            batch_size (int): how many samples per batch to load
            shuffle (bool):  set to True to have the data reshuffled at every epoch
            num_workers (int): how many subprocesses to use for data loading. 0 means that
                               the data will be loaded in the main process.
            optimizer: optimizer class from torch.optim
            optimizer_kwargs: optimizer keyword arguments
            lr_scheduler: one learing rate scheuler from torch.optim.lr_scheduler
            lr_scheduler_kwargs (dict): keyword arguments for lr_scheduler_class
            epochs (int): number of epochs
            earlystopping_kwargs (dict): Early stopping parameters. See gtorch_utils.models.callbacks.EarlyStopping class definition
            checkpoints (bool): whether or not work with checkpoint
            checkpoint_path (OrderedDict): Path to the checkpoint to load. You must provide at least
                                           the directory_path because it will be used to save new
                                           checkpoints. Items:
                directory_path: <path to the checkpoints directory>
                filename      : <checkpoint filename>.pth.tar
            saving_details (OrderedDict): Path to be used to load and save the best model found. Items:
                directory_path: <path to the best model directory>
                filename      : <best model filename>.pth
            tensorboard (bool): whether or not plot train and validation loss to tensorboard
        """
        self.cuda = kwargs.get('cuda', True)
        self.model = kwargs.get('model')
        self.sub_datasets = kwargs.get('sub_datasets', DB)
        self.dataset = kwargs.get('dataset')
        self.dataset_kwargs = kwargs.get('dataset_kwargs', {})
        self.batch_size = kwargs.get('batch_size', 6)
        self.shuffle = kwargs.get('shuffle', False)
        self.num_workers = kwargs.get('num_workers', 6)
        self.optimizer = kwargs.get('optimizer', optim.SGD)
        self.optimizer_kwargs = kwargs.get('optimizer_kwargs', dict(lr=1e-3, momentum=.9))
        self.lr_scheduler = kwargs.get('lr_scheduler', None)
        self.lr_scheduler_kwargs = kwargs.get('lr_scheduler_kwargs', {})
        self.epochs = kwargs.get('epochs')
        self.earlystopping_kwargs = kwargs.get('earlystopping_kwargs', dict(min_delta=1e-3, patience=8))
        self.checkpoints = kwargs.get('checkpoints', False)
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 10)
        self.checkpoint_path = kwargs.get(
            'checkpoint_path', OrderedDict(directory_path='tmp', filename=''))
        self.saving_details = kwargs.get(
            'saving_details', OrderedDict(directory_path='tmp', filename='best_model.pth'))
        self.tensorboard = kwargs.get('tensorboard', True)

        assert isinstance(self.cuda, bool)
        assert isinstance(self.model, nn.Module)
        assert isinstance(self.dataset_kwargs, dict)
        assert self.batch_size >= 1
        assert isinstance(self.shuffle, bool)
        assert self.num_workers >= 0
        assert isinstance(self.optimizer_kwargs, dict)
        assert isinstance(self.lr_scheduler_kwargs, dict)
        assert isinstance(self.epochs, int)
        assert self.epochs > 0
        assert isinstance(self.earlystopping_kwargs, dict)
        assert isinstance(self.checkpoints, bool)
        assert isinstance(self.checkpoint_interval, int)
        assert self.checkpoint_interval > 0
        assert isinstance(self.checkpoint_path, OrderedDict)
        assert isinstance(self.saving_details, OrderedDict)
        assert isinstance(self.tensorboard, bool)

        if self.checkpoint_path.get('filename', ''):
            assert self.checkpoint_path['filename'].endswith(self.checkpoint_extensions)
            assert os.path.isfile(os.path.join(*self.checkpoint_path.values()))

        assert self.saving_details['filename'].endswith(self.model_extensions)

        if self.cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.device == "cpu":
                warnings.warn("CUDA is not available. Using CPU")
        else:
            self.device = "cpu"

        self.model.to(self.device)

        self.train_loader = self.get_subdataset(self.sub_datasets.TRAIN)
        self.val_loader = self.get_subdataset(self.sub_datasets.VALIDATION)
        self.test_loader = self.get_subdataset(self.sub_datasets.TEST)

    def __call__(self):
        """ functor call """
        self.fit()
        self.predict()

    def save_checkpoint(self, epoch=0, optimizer=None, loss_logger=None):
        """
        Saves the model as a checkpoint for inference and/or resuming training

        Args:
            epoch (int): current epoch
            optimizer (self.optimizer): optimizer instance
            loss_logger (list): list with the tracked losses
        """
        assert isinstance(epoch, int)
        assert epoch >= 0
        assert isinstance(optimizer, self.optimizer)
        assert isinstance(loss_logger, list)

        if not os.path.isdir(self.checkpoint_path['directory_path']):
            os.makedirs(self.checkpoint_path['directory_path'])

        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_logger': loss_logger
        }
        torch.save(
            data,
            os.path.join(
                self.checkpoint_path['directory_path'],
                self.checkpoint_pattern.format(data['epoch'])
            )
        )

    def save(self):
        """ Saves the best model only for inference """
        if not os.path.isdir(self.saving_details['directory_path']):
            os.makedirs(self.saving_details['directory_path'])

        torch.save(self.model.state_dict(), os.path.join(*self.saving_details.values()))

    def get_subdataset(self, sub_dataset):
        """
        Returns a torch DataLoader for the espeficied sub dataset

        Args:
            sub_dataset ('str'): sub dataset type. See contants.DB class

        Returns:
            Dataloader
        """
        assert sub_dataset in self.sub_datasets.SUB_DATASETS
        dataset = self.dataset(sub_dataset, **self.dataset_kwargs)
        assert isinstance(dataset, BaseDataset)

        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def load_checkpoint(self, optimizer):
        """
        Loads the checkpoint for inference and/or resuming training

        Args:
            optimizer (self.optimizer): optimizer instance

        Returns:
            current epoch (int), loss_logger (list)
        """
        assert isinstance(optimizer, self.optimizer)

        chkpt_path = os.path.join(*self.checkpoint_path.values())
        assert os.path.isfile(chkpt_path)

        chkpt = torch.load(chkpt_path)
        self.model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])

        # sending model and optimizer to the right device
        self.model.to(self.device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        return chkpt['epoch'], chkpt['loss_logger']

    def load(self):
        """ Loads the best model for inference"""
        self.model.load_state_dict(torch.load(os.path.join(*self.saving_details.values())))

    @timing
    def fit(self):
        """
        Trains the model

        Args:
            print_mini_batches (int): number of mini batches to count before
                                      calculating and printing the loss
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_kwargs)
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)
        checkpoint = Checkpoint(self.checkpoint_interval)
        loss_logger = list()
        start_epoch = 0
        val_loss_min = np.inf

        # if a checkpoint file is provided, then load it
        if self.checkpoints and os.path.isfile(os.path.join(*self.checkpoint_path.values())):
            start_epoch, loss_logger = self.load_checkpoint(optimizer)
            val_loss_min = loss_logger[-1][1]
            # increasing to start at the next epoch
            start_epoch += 1

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        else:
            scheduler = None

        self.model.train()

        for epoch in range(start_epoch, self.epochs):
            train_loss = .0
            val_loss = .0

            ###################################################################
            #                             training                            #
            ###################################################################
            self.model.train()

            for data in self.train_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['feats'].to(self.device), data['label'].to(self.device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*inputs.size(0)

            if scheduler:
                scheduler.step()

            ###################################################################
            #                            validation                           #
            ###################################################################
            self.model.eval()

            with torch.no_grad():
                for data in self.val_loader:
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data['feats'].to(self.device), data['label'].to(self.device)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()*inputs.size(0)

            train_loss = train_loss/len(self.train_loader)
            val_loss = val_loss/len(self.val_loader)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, train_loss, val_loss))

            loss_logger.append((train_loss, val_loss, epoch))

            if self.checkpoints and checkpoint(epoch):
                self.save_checkpoint(epoch, optimizer, loss_logger)

            if earlystopping(val_loss, val_loss_min):
                break

            # save best model if validation loss has decreased
            if val_loss < val_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    val_loss_min, val_loss))
                self.save()
                val_loss_min = val_loss

        if self.tensorboard:
            PlotTensorBoard(loss_logger)()

        print('Training completed')

    @timing
    def predict(self):
        """ Tests the model and prints the accuracy achieved """
        correct = 0
        total = 0

        self.load()
        self.model.eval()

        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data['feats'].to(self.device), data['label'].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy: %d %%' % (100 * correct / total))
