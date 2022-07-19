# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/labels """

import os
from collections import namedtuple

from gutils.plot.tables import plot_color_table


__all__ = ['Detail', 'DatasetLabelsMixin']


Detail = namedtuple('Detail', ['colour', 'id', 'name', 'file_label', 'RGB'])


class DatasetLabelsMixin:
    """
    Contains the plot palette method for dataset label information classes

    Usage:
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from gtorch_utils.datasets.labels import Detail, DatasetLabelsMixin

        class DBLabels(DatasetLabelsMixin):
            MELANOCYTE = Detail('red', 1, 'Melanocyte', '', (255, 0, 0))
            HEPATOCYTE = Detail('green', 2, 'Hepatocyte', '', (0, 128, 0))
            MELANOPHAGE = Detail('Yellow', 3, 'Melanophage', '', (255, 255, 0))
            LABELS = (MELANOCYTE, HEPATOCYTE, MELANOPHAGE)
            CMAPS = tuple(
                LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
                for label in LABELS
            )

        DBLabels.plot_palette()
        plt.show()
    """

    @classmethod
    def plot_palette(cls, saving_path: str = ''):
        """
        Plots the colour palette and returns a matplotlib Figure.
        If a saving_path is provided, the palette is saved

        Args:
            saving_path <str>: path to the file to save the image. If not provided the palette
                               will not be saved. Default ''

        Usage:
            import matplotlib.pyplot as plt
            fig = <MyLabelClass>.plot_palette('<path to my director>my_palettet.png')
            plt.show()

        Returns:
            figure <matplotlib.figure.Figure>
        """
        assert isinstance(saving_path, str), type(saving_path)

        label_colours = {}

        for label in cls.LABELS:
            label_colours[label.name] = [c/255 for c in label.RGB]

        fig = plot_color_table(label_colours, "Label colours")

        if saving_path:
            dirname = os.path.dirname(saving_path)

            if dirname and not os.path.isdir(dirname):
                os.makedirs(dirname)

            fig.savefig(saving_path)

        return fig
