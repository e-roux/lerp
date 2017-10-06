# -*- coding: utf-8 -*-
"""
This module delivers utilities to manipulate data meshes

"""

import lerp.core.config_init
from lerp.core.config import (get_option, set_option, reset_option,
                              describe_option, option_context, options)

from lerp.mesh import BreakPoints, mesh1d, mesh2d, mesh3d, mesh4d, mesh5d
from lerp.polymesh import polymesh2d, polymesh3d


__version__ = "0.1aN"

# Attention, utilisation d'ascii pour les chaînes de caractères
__all__ = ["mesh1d", "mesh2d", "mesh3d", "mesh4d", "mesh5d",
           "BreakPoints", "polymesh2d", "polymesh3d"]
