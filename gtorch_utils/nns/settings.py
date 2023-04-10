# -*- coding: utf-8 -*-
""" gtorch_utils/nns/managers/settings.py """

try:
    import settings
except ModuleNotFoundError:
    settings = None

USE_AMP = settings.USE_AMP if hasattr(settings, 'USE_AMP') else False
DISABLE_PROGRESS_BAR = settings.DISABLE_PROGRESS_BAR if hasattr(settings, 'DISABLE_PROGRESS_BAR') else False
