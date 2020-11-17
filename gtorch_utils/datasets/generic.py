# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/generic """

from torch.utils.data import Dataset

from gtorch_utils.constants import DB
# from gutils.numpy_.numpy_ import LabelMatrixManager
# from sklearn.model_selection import train_test_split


class BaseDataset(Dataset):
    """ Base dataset template """

    def __init__(self, subset, **kwargs):
        """ Initialized the instance """

        assert subset in DB.SUB_DATASETS
        self.subset = subset

        self.initialization(subset, **kwargs)

    def initialization(self, subset, **kwargs):
        """
        Operations required to enable the retrieval of feats and labels for the
        subset (train, val, or test)
        """
        # e.g. ################################################################
        # dbhandler = kwargs.get('dbhandler')
        # normalizer = kwargs.get('normalizer')
        # val_size = kwargs.get('val_size')

        # if self.subset in (DB.TRAIN, DB.VALIDATION):
        #     self.feats, self.labels = dbhandler(normalizer)()[:2]
        #     self.labels = LabelMatrixManager.get_1d_array_from_2d_matrix(self.labels)

        #     if subset == DB.TRAIN:
        #         self.feats, _, self.labels, _ = train_test_split(
        #             self.feats.T, self.labels, test_size=val_size, random_state=self.random_state,
        #             stratify=self.labels
        #         )
        #     else:
        #         _, self.feats, _,  self.labels = train_test_split(
        #             self.feats.T, self.labels, test_size=val_size, random_state=self.random_state,
        #             stratify=self.labels
        #         )

        #     self.feats = self.feats.T

        # else:
        #     self.feats, self.labels = dbhandler(normalizer)()[2:]
        #     self.labels = LabelMatrixManager.get_1d_array_from_2d_matrix(self.labels)

        # self.feats = self.feats.astype(np.float32)

        raise NotImplementedError

    def __len__(self):
        """
        Returns:
            dataset size (int)
        """
        # e.g. ################################################################
        # return self.labels.shape[0]

        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns:
            dict(feats=..., label=...)
        """
        # e.g. ################################################################
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # # feats = self.get_default_data_transforms()[self.subset](self.feats[:, idx].ravel())
        # # feats = self.get_default_data_transforms()[self.subset](
        # #     self.feats[:, idx].ravel()[np.newaxis, np.newaxis, :])

        # return dict(
        #     feats=torch.from_numpy(self.feats[:, idx].ravel()),
        #     label=self.labels[idx]
        # )

        raise NotImplementedError
