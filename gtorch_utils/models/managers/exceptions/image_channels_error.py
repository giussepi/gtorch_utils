# -*- coding: utf-8 -*-
""" gtorch_utils/models/managers/exceptions/image_channels_error """


class ModelMGRImageChannelsError(RuntimeError):
    """
    Exception to be raised when the number of input channels defined in model is
    different than number of channels in the loaded image
    """

    def __init__(self, model_n_channels, img_channels):
        """
        Initializes the instance with a custom message

        Args:
            model_n_channels <int>: number of channels defined in the model
            img_channels     <int>: number of channels in the image
        """
        assert isinstance(model_n_channels, int), type(model_n_channels)
        assert isinstance(img_channels, int), type(img_channels)

        super().__init__(
            f'The network has been defined with {model_n_channels} input channels, '
            f'but loaded images have {img_channels} channels. Please check that '
            'the images are loaded correctly.'
        )
