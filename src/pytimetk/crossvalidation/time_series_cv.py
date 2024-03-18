import pandas as pd
import numpy as np


class TimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes. Includes a shift for each test set."""

    def __init__(
        self,
        n_splits=3,
        train_period_length=126,
        test_period_length=21,
        lookahead=None,
        shift_length=0,  # New parameter to specify the shift length
        date_idx='date',
        shuffle=False,
        seed=None,
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shift_length = shift_length  # Store the shift length
        self.shuffle = shuffle
        self.seed = seed
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        
        splits = []
        for i in range(self.n_splits):
            # Adjust the end index for the test set to include the shift for subsequent splits
            test_end_idx = i * self.test_length + i * self.shift_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            
            if train_start_idx >= len(days):
                break  # Break if the start index goes beyond the available data
            
            dates = X.reset_index()[[self.date_idx]]
            train_idx = dates[(dates[self.date_idx] > days[min(train_start_idx, len(days)-1)])
                              & (dates[self.date_idx] <= days[min(train_end_idx, len(days)-1)])].index
            test_idx = dates[(dates[self.date_idx] > days[min(test_start_idx, len(days)-1)])
                             & (dates[self.date_idx] <= days[min(test_end_idx, len(days)-1)])].index
            
            if self.shuffle:
                if self.seed is not None:
                    np.random.seed(self.seed)
                
                train_idx_list = list(train_idx)
                np.random.shuffle(train_idx_list)
                train_idx = np.array(train_idx_list)
            else:
                train_idx = train_idx.to_numpy()
                
            test_idx = test_idx.to_numpy()
            
            splits.append((train_idx, test_idx))
        
        return splits

    def get_n_splits(self, X=None, y=None, groups=None):
        """Adjusts the number of splits if there's not enough data for the desired configuration."""
        return self.n_splits
    
    