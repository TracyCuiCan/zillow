import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# customized setting
from common import DATA_TRAIN, DATA_TEST, DATA_PROP, DATA_ID, DATA_OUT_FEATURE
from common import SUBMIT_ID, SUBMISSION_FOLDER

class DataWarehouse():
    """A class to handle data IO
    """

    def __init__(self):
        pass

    def read_data(self):
        """Read in train and test data. If the data is in multiple files, need more work to separate data and ID.
        """
        print("Loading data...")
        train = pd.read_csv(DATA_TRAIN)
        test = pd.read_csv(DATA_TEST)
        test['parcelid'] = test['ParcelId']
        prop = pd.read_csv(DATA_PROP)

        print('Binding to float32...')

        for c, dtype in zip(prop.columns, prop.dtypes):
            if dtype == np.float64:
                prop[c] = prop[c].astype(np.float32)
            prop[c] = prop[c].fillna(-1)
            if dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(prop[c].values))
                prop[c] = lbl.transform(list(prop[c].values))


        print("Joining train and test with property...")
        self.train_in = train.merge(prop, how='left', on='parcelid')
        self.test_in = test.merge(prop, how='left', on='parcelid')
        # get response variable and id
        self.train_out = self.train_in[DATA_OUT_FEATURE]
        self.train_id = self.train_in[DATA_ID]

        # remove unnecessary information
        del self.train_in[DATA_ID]
        del self.train_in[DATA_OUT_FEATURE]

        # remove unnecessary information from test
        self.test_id = self.test_in[SUBMIT_ID]
        #del self.test_in[SUBMIT_ID]

    def generate_submission(self, ypred):
        sub = pd.read_csv(DATA_TEST)
        for c in sub.columns[sub.columns != 'ParcelId']:
            sub[c] = ypred

        print('writing csv...')
        sub.to_csv(os.path.join(SUBMISSION_FOLDER, "submission.csv"), index=False, float_format='%.4f')

