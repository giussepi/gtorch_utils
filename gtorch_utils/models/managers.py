# -*- coding: utf-8 -*-
""" gtorch_utils/models/managers """

import os
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gutils.decorators import timing
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from gtorch_utils.constants import DB
from gtorch_utils.datasets.generic import BaseDataset
from gtorch_utils.models.callbacks import EarlyStopping


class ModelMGR:
    """
    General model managers

    Usage:
        ModelMGR(
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
            lr_scheduler_kwargs=None,
            epochs=200,
            earlystopping_kwargs=dict(min_delta=1e-2, patience=8),
            saving_details=OrderedDict(directory_path='tmp', filename='mymodel.pth')
        )()
    """

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
            saving_details (OrderedDict): Contains data to properly save the model. Items:
                'directory_path': <path to the directory containing the model weights>
                'filename'      : <name of the file containing the model weights>.pth
        """
        self.cuda = kwargs.get('cuda', True)
        self.model = kwargs.get('model')
        self.sub_datasets = kwargs.get('sub_datasets', DB)
        self.dataset = kwargs.get('dataset')
        self.dataset_kwargs = kwargs.get('dataset_kwargs', {})
        self.batch_size = kwargs.get('batch_size', 1)  # 6
        self.shuffle = kwargs.get('shuffle', False)
        self.num_workers = kwargs.get('num_workers', 0)  # 12
        self.optimizer = kwargs.get('optimizer', optim.SGD)
        self.optimizer_kwargs = kwargs.get('optimizer_kwargs', {})
        self.lr_scheduler = kwargs.get('lr_scheduler', None)
        self.lr_scheduler_kwargs = kwargs.get('lr_scheduler_kwargs', {})
        self.epochs = kwargs.get('epochs', 200)
        self.earlystopping_kwargs = kwargs.get('earlystopping_kwargs', dict(min_delta=1e-2, patience=8))
        self.saving_details = kwargs.get(
            'saving_details', OrderedDict(directory_path='tmp', filename='mymodel.pth'))

        assert isinstance(self.cuda, bool)
        assert isinstance(self.model, nn.Module)
        assert isinstance(self.dataset_kwargs, dict)
        assert self.batch_size >= 1
        assert isinstance(self.shuffle, bool)
        assert self.num_workers >= 0
        assert isinstance(self.optimizer_kwargs, dict)
        assert isinstance(self.lr_scheduler_kwargs, dict)
        assert self.epochs > 0
        assert isinstance(self.earlystopping_kwargs, dict)
        assert isinstance(self.saving_details, OrderedDict)

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

    def __call__(self, **kwargs):
        """ functor call """
        self.fit()
        self.predict()

    def save(self):
        """ Saves the model """
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

    def load(self):
        """ Loads the model """
        self.model.load_state_dict(torch.load(os.path.join(*self.saving_details.values())))

    @timing
    def fit(self):
        """
        Trains the model

        Args:
            print_mini_batches (int): number of mini batches to count before
                                      calculating and printing the loss
        """
        writer = SummaryWriter()
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_kwargs)
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        else:
            scheduler = None

        self.model.train()

        val_loss_min = np.inf

        for epoch in range(self.epochs):

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

            writer.add_scalars(
                'loss: Train & Validation', {'Train': train_loss, 'Validation': val_loss}, epoch)

            if earlystopping(val_loss, val_loss_min):
                break

            # save model if validation loss has decreased
            if val_loss < val_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    val_loss_min, val_loss))
                self.save()
                val_loss_min = val_loss

        writer.close()
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
