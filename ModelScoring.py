import pandas as pd
import pickle

def PredictBuySell(Inputfile, modelLoc):
    dfStocks = pd.read_csv(Inputfile)
    dfStocks['avgMVolRatio'] = dfStocks['tottrdqty']/dfStocks['avgMvol']
    dfStocks['avgDVolRatio'] = dfStocks['tottrdqty']/dfStocks['preVolume']
    dfStocks['avgWVolRatio'] = dfStocks['tottrdqty']/dfStocks['avgWvol']
    dfStocks['avg50DVolRatio'] = dfStocks['tottrdqty']/dfStocks['avgD50vol']
    dfStocks['avgMPRatio'] = dfStocks['last']/dfStocks['avgMPrice']
    dfStocks['avgDPRatio'] = dfStocks['last']/dfStocks['avgDPrice']
    dfStocks['avgWPRatio'] = dfStocks['last']/dfStocks['avgWPrice']
    dfStocks['avg50DPRatio'] = dfStocks['last']/dfStocks['avgD50Price']
    
    dfStocksTarget = dfStocks['buyflag']
    X_test = dfStocks.drop(columns=['id', 'symbol', 'buyflag', 'sellflag', 'zTransitDate', 'sTransitDate', 'pp', 'rsiTransitDate', 'transitdate', 'timestamp', 'exchangename', 'exchangeid', 'priceTrend', 'cciTransitDate', 'crossDate5c13', 'crossDate5c26', 'crossDate50c200'])
    loaded_model = pickle.load(open(modelLoc + '\\RandomForestModel', 'rb'))
    result = loaded_model.predict(X_test)
    
    dfStocks['Predicted'] = result
    dfStocks.to_csv(modelLoc + '\\PredictedCSV.csv')
    
    return 'predicted results are stored at location ' + modelLoc + '\PredictedCSV.csv'

PredictBuySell('E:\software\MachineLearning\Saral\StocksData_9Aug.csv', 'E:\software\MachineLearning\Saral')