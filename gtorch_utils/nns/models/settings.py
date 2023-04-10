# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/settings.py """

try:
    import settings
except ModuleNotFoundError:
    settings = None

USE_AMP = settings.USE_AMP if hasattr(settings, 'USE_AMP') else False
