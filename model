import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller 
from sklearn.linear_model import LinearRegression

class CointModel(object):
    '''
    CointModel class for Engle-Granger 2-steps Cointegration Screening 
    '''
    
    def __init__(self):
        self.residual_lag = 1
    
    def initialise_model(self, data, sig_lvl):
        '''
        Function to initialise a CointModel object 
        
        Parameters:
        -----------
        data: dictionary of bql fields 
        sig_lvl: significance level for cointegration test
        '''
        
        self.data = data
        self.sig_lvl = sig_lvl
    
    def adf_test(self, time_series, max_lag=None):
        '''
        Function to carry out Augmented Dickey–Fuller (ADF) test for stationarity 
        
        Parameters:
        -----------
        time series: time series data 
        
        Returns
        -------
        adf result, p-value, OLS regression model, critical values, test statistics 
        '''
        
        if time_series.isna().sum() > 0:
            return False, None, None, None, None 
        else:
            adf_res = adfuller(time_series, maxlag=max_lag, regresults=True)  
            adf_pass = True if adf_res[0] < adf_res[2].get(self.sig_lvl) else False
            return adf_pass,  adf_res[1],  adf_res[3], adf_res[2], adf_res[0]
    
    def residual_adf(self, ticker1_log_price, ticker2_log_price): 
        '''
        Function to carry out Augmented Dickey–Fuller (ADF) test on residual series 
        
        Parameters:
        -----------
        ticker1_log_price: log price series of 1st ticker 
        ticker2_log_price: log price series of 2nd ticker 
        
        Returns
        -------
        adf result, p-value, hedge ratio, residual regression model, residual, critical values, test statistics 
        '''
        
        lm_model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
        lm_model.fit(ticker1_log_price.values.reshape(-1,1), ticker2_log_price.values)
        y_fit = lm_model.coef_ * ticker1_log_price.values + lm_model.intercept_
        y_residual = pd.Series(ticker2_log_price.values - y_fit)
        adf_pass, p_val, reg_results, adf_stat, test_stat = self.adf_test(y_residual, self.residual_lag)
        return adf_pass, p_val, lm_model.coef_[0], reg_results, y_residual, adf_stat, test_stat
    
    def half_life(self, reg_results):
        '''
        Function to compute the half-life of cointegrated pair 
        
        Parameters:
        -----------
        reg_results: residual regression model 
        
        Returns
        -------
        half-life  
        '''
        
        lambda_val = reg_results.resols.params[0]
        half_life = -np.log(2)/ lambda_val
        return half_life
    
    def coint_test(self, ticker1_id, ticker2_id):
        '''
        Function to carry out Engle-Granger Cointegration test for a pair of tickers 
        
        Parameters:
        -----------
        ticker1_id: name of 1st ticker 
        ticker2_id: name of 2nd ticker 
        
        Returns
        -------
        cointegration result, tuple of cointegrated pair's key info 
        '''
        
        adf_ticker1_first_diff, _, _, _, _ = self.adf_test(self.data.get('first_diff')[ticker1_id])
        adf_ticker2_first_diff, _, _, _, _ = self.adf_test(self.data.get('first_diff')[ticker2_id])
        
        if adf_ticker1_first_diff == True and adf_ticker2_first_diff == True:
            ticker1_log_price = self.data.get('log_price')[ticker1_id]
            ticker2_log_price = self.data.get('log_price')[ticker2_id]
            adf_pass1, p_val1, hedge_ratio1, reg_results1, spread1, adf_stat1, test_stat1 = self.residual_adf(ticker1_log_price, ticker2_log_price)
            adf_pass2, p_val2, hedge_ratio2, reg_results2, spread2, adf_stat2, test_stat2 = self.residual_adf(ticker2_log_price, ticker1_log_price)

            if adf_pass1 and adf_pass2:
                if p_val1 < p_val2:
                    self.residual_stats[(ticker1_id, ticker2_id)] = adf_stat1
                    self.residual_stats[(ticker1_id, ticker2_id)]['Test Stat'] = test_stat1
                    return True, (ticker1_id, ticker2_id, hedge_ratio1, self.half_life(reg_results1), spread1)
                else:
                    self.residual_stats[(ticker2_id, ticker1_id)] = adf_stat2
                    self.residual_stats[(ticker2_id, ticker1_id)]['Test Stat'] = test_stat2
                    return True, (ticker2_id, ticker1_id, hedge_ratio2, self.half_life(reg_results2), spread2)
        return False, None
    
    def screen_univ(self):
        '''
        Function to carry out Cointegration screening for an universe 
        
        Returns
        -------
        dictionary of cointegrated pairs
        '''
        
        univ_tickers = self.data.get('univ_tickers')
        lst_pairs = [(univ_tickers[i], univ_tickers[j]) for i in range(len(univ_tickers)) for j in range(i+1,len(univ_tickers))]
        
        self.coint_pairs, self.residual_stats = {}, {}
        for pair in lst_pairs:
            ticker1_id, ticker2_id = pair[0], pair[1]
            coint_pass, coint_res = self.coint_test(ticker1_id, ticker2_id)
            
            if coint_pass:
                self.coint_pairs[(coint_res[0], coint_res[1])] = [coint_res[2], coint_res[3], coint_res[4]]
        
        return self.coint_pairs 
