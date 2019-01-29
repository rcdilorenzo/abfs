from collections import namedtuple as Struct
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

DataSplitConfig = Struct('DataSplitConfig', ['validation_size', 'test_size', 'random_seed'])

DEFAULT_SPLIT_CONFIG = DataSplitConfig(0.2, 0.2, 1337)

class GroupDataSplit():
    def __init__(self, df, key, config=DEFAULT_SPLIT_CONFIG):
        self.config = config
        self.key = key
        self._df = df
        self._split_data()

    @property
    def total(self):
        """Total records in the data frame"""
        return len(self._df)

    def train_df(self):
        """Randomized train data frame"""
        return self._train_df.sample(frac=1).reset_index(drop=True)

    @property
    def val_df(self):
        """Validation data frame"""
        return self._val_df

    @property
    def test_df(self):
        """Test data frame"""
        return self._test_df

    @property
    def test_split(self):
        return GroupShuffleSplit(test_size=self.config.test_size,
                                 random_state=self.config.random_seed).split

    @property
    def val_split(self):
        val_size = self.config.validation_size / (1 - self.config.test_size)
        return GroupShuffleSplit(test_size=val_size,
                                 random_state=self.config.random_seed).split

    def _split_data(self):
        rem_indices, test_indices = next(
            self.test_split(self._df, groups=self._df[self.key])
        )

        rem_df = self._df.iloc[rem_indices]
        train_indices, val_indices = next(
            self.val_split(rem_df, groups=rem_df[self.key])
        )

        self._test_df = self._df.iloc[test_indices]
        self._val_df = rem_df.iloc[val_indices]
        self._train_df = rem_df.iloc[train_indices]
