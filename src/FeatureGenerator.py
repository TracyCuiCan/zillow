import logging, yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureGenerator(object):
    '''
    A class to extract feature from data. Try to spend more time on feature engineering. It gives a lot of improvement.
    This feature generateor uses Kaggle-Titanic data for illustration. Most of time are spent in feature extraction.
    '''
    def __init__(self, datawarehouse, config):
        self._DW = datawarehouse
        self.config = config
        self.feature_names = []

    def compute_features(self):
        logging.info('Start to compute features')

        # get basic features
        self.fTrain = []
        self.fTest = []

        # append features as new columns
        fconfig = self.config['features']

        for k,v in fconfig.items():
            if(v!=-1):
                self.feature_names.append(k)
                opt = v
                tmpTrain, tmpTest = eval('self.extract_%s(%s)'%(k,'opt'))
                logging.info('processing %s: added %d columns'%(k, tmpTrain.shape[1]))

                if(len(self.fTrain)):
                    self.fTrain = np.hstack((self.fTrain, tmpTrain))
                    self.fTest = np.hstack((self.fTest, tmpTest))
                else:
                    self.fTrain = tmpTrain
                    self.fTest = tmpTest

        logging.info("finish feature extraction")

    @property
    def train_out(self):
        return self.DW.train_out

    @property
    def DW(self):
        return self._DW

    @DW.setter
    def DW(self, value):
        raise Exception('Can not modify DataWarehouse through FeatureGenerator')

    def extract_lotsizesquarefeet(self, opt):
        data_all = pd.concat([self.DW.train_in['lotsizesquarefeet'], self.DW.test_in['lotsizesquarefeet']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest

    def extract_taxamount(self, opt):
        # impute date
        data_all = pd.concat([self.DW.train_in['taxamount'], self.DW.test_in['taxamount']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest


    def normalize_features(self):
        # normalize feature matrices
        self.scaler = StandardScaler()
        self.scaler.fit(self.fTrain)
        self.fTrain = self.scaler.transform(self.fTrain)
        self.fTest = self.scaler.transform(self.fTest)
