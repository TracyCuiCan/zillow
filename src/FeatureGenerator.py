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

    def extract_taxamount(self, opt):
        data_all = pd.concat([self.DW.train_in['taxamount'], self.DW.test_in['taxamount']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        #return self.DW.train_in['taxamount'].as_matrix()[:,None], self.DW.test_in['taxamount'].as_matrix()[:, None]

    def extract_longitude(self, opt):
        #TODO-Change to cluster
        return self.DW.train_in['longitude'].as_matrix()[:,None], self.DW.test_in['longitude'].as_matrix()[:, None]

    def extract_latitude(self, opt):
        #TODO-Change to cluster
        return self.DW.train_in['latitude'].as_matrix()[:,None], self.DW.test_in['latitude'].as_matrix()[:, None]

    def extract_yearbuilt(self, opt):
        return self.DW.train_in['yearbuilt'].as_matrix()[:,None], self.DW.test_in['yearbuilt'].as_matrix()[:, None]

    def extract_calculatedfinishedsquarefeet(self, opt):
        '''
        data_all = pd.concat([self.DW.train_in['calculatedfinishedsquarefeet'], self.DW.test_in['calculatedfinishedsquarefeet']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        '''
        return self.DW.train_in['calculatedfinishedsquarefeet'].as_matrix()[:,None], self.DW.test_in['calculatedfinishedsquarefeet'].as_matrix()[:, None]

    def extract_structuretaxvaluedollarcnt(self, opt):
        '''
        data_all = pd.concat([self.DW.train_in['structuretaxvaluedollarcnt'], self.DW.test_in['structuretaxvaluedollarcnt']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        '''
        return self.DW.train_in['structuretaxvaluedollarcnt'].as_matrix()[:,None], self.DW.test_in['structuretaxvaluedollarcnt'].as_matrix()[:, None]

    def extract_landtaxvaluedollarcnt(self, opt):
        return self.DW.train_in['landtaxvaluedollarcnt'].as_matrix()[:,None], self.DW.test_in['landtaxvaluedollarcnt'].as_matrix()[:, None]

    def extract_taxvaluedollarcnt(self, opt):
        return self.DW.train_in['taxvaluedollarcnt'].as_matrix()[:,None], self.DW.test_in['taxvaluedollarcnt'].as_matrix()[:, None]
    
    def extract_lotsizesquarefeet(self, opt):
        data_all = pd.concat([self.DW.train_in['lotsizesquarefeet'], self.DW.test_in['lotsizesquarefeet']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        #return self.DW.train_in['lotsizesquarefeet'].as_matrix()[:,None], self.DW.test_in['lotsizesquarefeet'].as_matrix()[:, None]

    def extract_regionidzip(self, opt):
        return self.DW.train_in['regionidzip'].as_matrix()[:,None], self.DW.test_in['regionidzip'].as_matrix()[:, None]

    def extract_rawcensustractandblock(self, opt):
        return self.DW.train_in['rawcensustractandblock'].as_matrix()[:,None], self.DW.test_in['rawcensustractandblock'].as_matrix()[:, None]

    def extract_finishedsquarefeet12(self, opt):
        '''
        data_all = pd.concat([self.DW.train_in['finishedsquarefeet12'], self.DW.test_in['finishedsquarefeet12']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        '''
        return self.DW.train_in['finishedsquarefeet12'].as_matrix()[:,None], self.DW.test_in['finishedsquarefeet12'].as_matrix()[:, None]

    def extract_bathroomcnt(self, opt):
        '''
        data_all = pd.concat([self.DW.train_in['bathroomcnt'], self.DW.test_in['bathroomcnt']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        '''
        return self.DW.train_in['bathroomcnt'].as_matrix()[:,None], self.DW.test_in['bathroomcnt'].as_matrix()[:, None]

    def extract_finishedsquarefeet15(self, opt):
        '''
        tmpTrain = self.DW.train_in['finishedsquarefeet15'].apply(lambda x: 0 if x=='NaN' else 1).as_matrix()[:,None]
        tmpTest = self.DW.test_in['finishedsquarefeet15'].apply(lambda x: 0 if x=='NaN' else 1).as_matrix()[:,None]
        return tmpTrain, tmpTest
        '''
        return self.DW.train_in['finishedsquarefeet15'].as_matrix()[:,None], self.DW.test_in['finishedsquarefeet15'].as_matrix()[:, None]

    def extract_bedroomcnt(self, opt):
        '''
        data_all = pd.concat([self.DW.train_in['bedroomcnt'], self.DW.test_in['bedroomcnt']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        '''
        return self.DW.train_in['bedroomcnt'].as_matrix()[:,None], self.DW.test_in['bedroomcnt'].as_matrix()[:, None]

    def extract_regionidneighborhood(self, opt):
        #TODO-change to cluster
        return self.DW.train_in['regionidneighborhood'].as_matrix()[:,None], self.DW.test_in['regionidneighborhood'].as_matrix()[:, None]

    def extract_censustractandblock(self, opt):
        #This feature has enough data, also it's not a number format
        return self.DW.train_in['censustractandblock'].as_matrix()[:,None], self.DW.test_in['censustractandblock'].as_matrix()[:, None]

    def extract_regionidcity(self, opt):
        #TODO-Change to cluster
        return self.DW.train_in['regionidcity'].as_matrix()[:,None], self.DW.test_in['regionidcity'].as_matrix()[:, None]


    def extract_buildingqualitytypeid(self, opt):
        return self.DW.train_in['buildingqualitytypeid'].as_matrix()[:,None], self.DW.test_in['buildingqualitytypeid'].as_matrix()[:, None]

    def extract_garagetotalsqft(self, opt):
        '''
        data_all = pd.concat([self.DW.train_in['garagetotalsqft'], self.DW.test_in['garagetotalsqft']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest
        '''
        return self.DW.train_in['garagetotalsqft'].as_matrix()[:,None], self.DW.test_in['garagetotalsqft'].as_matrix()[:, None]

    def normalize_features(self):
        # normalize feature matrices
        self.scaler = StandardScaler()
        self.scaler.fit(self.fTrain)
        self.fTrain = self.scaler.transform(self.fTrain)
        self.fTest = self.scaler.transform(self.fTest)
