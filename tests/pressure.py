# -*- coding: utf-8 -*-

import pandas as pd
from lerp import Mesh


class PyDataSet(object):

    """
    Retrieves R Dataset
    """
    db = pd.read_csv(
        "http://vincentarelbundock.github.com/Rdatasets/datasets.csv")
    db.set_index('Item', drop=True, inplace=True)

    @classmethod
    def search(cls, name):
        """
        Search a dataset """
        # if name.isalpha():
        #    name = name.lower()
        return cls.db[cls.db.Title.str.contains(name, case=False) |
                      cls.db.index.str.contains(name, case=False)]

    @classmethod
    def get(cls, name):
        assert name in cls.db.Title, f"No dataset named {name}"
        df = pd.read_csv(cls.db.loc[name].csv)
        df.set_index(df.columns[0], drop=True, inplace=True)
        df.index.name = 'Sample'
        return df

    @classmethod
    def list(cls):
        return cls.db


pressure = PyDataSet.get('pressure')
pressure.plot('temperature', 'pressure')
pm = Mesh(*pressure.values.T)

# print(pm)
