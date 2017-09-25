"""
This module is imported from the pandas package __init__.py file
in order to ensure that the core.config options registered here will
be available as soon as the user loads the package. if register_option
is invoked inside specific modules, they will not be registered until that
module is imported, which may or may not be a problem.

If you need to make sure options are available even before a certain
module is imported, register them here rather then in the module.

License
=======
From pandas.core.config.py

BSD 3-Clause License

Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc.
and PyData Development Team
All rights reserved.

"""
import lerp.core.config as cf


pc_max_seq_items = """
: int or None
    when pretty-printing a long sequence, no more then `max_seq_items`
    will be printed. If items are omitted, they will be denoted by the
    addition of "..." to the resulting string.

    If set to None, the number of items to be printed is unlimited.
"""


style_backup = dict()


with cf.config_prefix('display'):
    cf.register_option('max_seq_items', 100, pc_max_seq_items)
