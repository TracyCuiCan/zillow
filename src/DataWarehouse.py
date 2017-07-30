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


        train = train[train.logerror > -0.4] 
        train = train[train.logerror < 0.419]

        print("Joining train and test with property...")
        train = train.merge(prop, how='left', on='parcelid')
        self.train_out = train[DATA_OUT_FEATURE]
        self.train_id = train[DATA_ID]

        self.train_in = train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc','propertycountylandusecode'], axis=1)
        training_columns = self.train_in.columns

        test = test.merge(prop, how='left', on='parcelid')
        self.test_id = test[SUBMIT_ID]
        self.test_in = test[training_columns]
        print('After removing outliers:')     
        print('Shape train: {}\nShape test: {}'.format(self.train_in.shape, self.test_in.shape))
        

    def generate_submission(self, ypred):
        sub = pd.read_csv(DATA_TEST)
        for c in sub.columns[sub.columns != 'ParcelId']:
            sub[c] = ypred

        print('writing csv...')
        sub.to_csv(os.path.join(SUBMISSION_FOLDER, "submission.csv"), index=False, float_format='%.4f')

