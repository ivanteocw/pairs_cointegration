import pandas as pd
import math, statistics

class BacktestingModel(object):
    '''
    BacktestingModel class for computation of backtesting metrics 
    '''
        
    def initialise_model(self, coint_pairs, data, init_cap, std):
        '''
        Function to initialise a BacktestingModel object 
        
        Parameters:
        -----------
        coint_pairs: dictionary of cointegrated pairs
        data: dictionary of bql fields 
        init_cap: floating value of initial capital 
        std: absolute value of standard deviation for trading signals generation 
        '''
        
        self.coint_pairs = coint_pairs
        self.data = data
        self.init_cap = init_cap 
        self.std = std
        self.zscores = self.get_zscores()  
        
    def get_zscores(self):
        '''
        Function to get residual z-scores for co-integrated pairs  
        
        Returns
        -------
        dictionary of z-score series 
        '''
        
        zscores = {}
        for key_pair, coint_res in self.coint_pairs.items():
            lookback = math.ceil(coint_res[1])
            spread_window = coint_res[2].rolling(window=lookback)
            spread_avg = spread_window.mean()
            spread_std = spread_window.std()
            zscore = (coint_res[2] - spread_avg)/ spread_std 
            zscores[key_pair] = zscore
        return zscores 
    
    def get_spread(self, indep_price, dep_price, hedge_ratio):
        '''
        Function to get spread between the independent and dependent tickers
        
        Parameters:
        -----------
        indep_price: traded price of independent ticker 
        dep_price: traded price of dependent ticker 
        hedge_ratio: ratio between the tickers 
        
        Returns
        -------
        spread 
        '''
        
        spread = dep_price - (indep_price * hedge_ratio)
        return spread    

    def get_pnl(self, start_pos, start_spread, end_spread, start_num_spread):
        '''
        Function to get pnl for a trade 
        
        Parameters:
        -----------
        start_pos: starting trade position (long: 1, short: -1)
        start_spread: value of starting spread
        end_spread: value of ending spread 
        start_num_spread: number of spread 
        
        Returns
        -------
        pnl 
        '''
        
        if start_pos == 1:
            return (end_spread - start_spread) * start_num_spread
        else:
            return (start_spread - end_spread) * start_num_spread
    
    def run(self):
        '''
        Function to carry out backtesting for all cointegrated pairs 
        
        Returns
        -------
        dictionary of key backtesting metrics/ results 
        '''

        trade_dates, pnl_vals, cap_vals, all_spreads = {}, {}, {}, {}
        dates = self.data.get('dates') 
        
        for key_pair, coint_res in self.coint_pairs.items():
            indep_ticker_id, dep_ticker_id = key_pair[0], key_pair[1]
            indep_price = self.data.get('price')[indep_ticker_id].tolist()
            dep_price = self.data.get('price')[dep_ticker_id].tolist()
            hedge_ratio = coint_res[0]
            zscore = self.zscores.get(key_pair).tolist()
            trade_date, pnl_val, cap_val, all_spread = [], [], [self.init_cap, ], []
            start_spread, start_price, start_pos, start_num_spread, start_cap, start_date = None, None, 0, None, self.init_cap, None 
            
            for i in range(len(zscore)):
                # Stop trading if capital becomes negative
                if start_cap <= 0:
                    break
                # Last Trading Day - close the position 
                if i == (len(zscore)-1) and start_pos != 0:
                    end_spread = self.get_spread(indep_price[i], dep_price[i], hedge_ratio)
                    pnl = self.get_pnl(start_pos, start_spread, end_spread, start_num_spread)
                    start_cap = start_cap + pnl
                    trade_date.append((start_date, dates[i]))
                    pnl_val.append(pnl)
                    cap_val.append(start_cap) 
                    start_pos = 0
                # Open a position - buy or long the spread 
                elif start_pos == 0 and zscore[i] < -self.std:
                    start_spread = self.get_spread(indep_price[i], dep_price[i], hedge_ratio)
                    start_pos = 1 
                    start_num_spread = start_cap/abs(start_spread)
                    start_date = dates[i]
                # Open a position - sell the spread 
                elif start_pos == 0 and zscore[i] > self.std:
                    start_spread = self.get_spread(indep_price[i], dep_price[i], hedge_ratio)
                    start_pos = -1 
                    start_num_spread = start_cap/abs(start_spread)
                    start_date = dates[i]
                # Take profit immediately once mean reverted 
                elif (start_pos != 0 and abs(zscore[i]) < 0.5) or (start_pos == 1 and zscore[i] > 0) or (start_pos == -1 and zscore[i] < 0):
                    end_spread = self.get_spread(indep_price[i], dep_price[i], hedge_ratio)
                    pnl = self.get_pnl(start_pos, start_spread, end_spread, start_num_spread)
                    start_cap = start_cap + pnl
                    trade_date.append((start_date, dates[i]))
                    pnl_val.append(pnl)
                    cap_val.append(start_cap)
                    start_pos = 0
                    
                all_spread.append(self.get_spread(indep_price[i], dep_price[i], hedge_ratio))
                    
            trade_dates[key_pair] = trade_date
            pnl_vals[key_pair] = pnl_val
            cap_vals[key_pair] = cap_val
            all_spreads[key_pair] = all_spread
            
        self.trade_dates = trade_dates 
        self.pnl_vals = pnl_vals
        self.cap_vals = cap_vals
        self.all_spreads = all_spreads
        
        return {
            'trade_dates': self.trade_dates,
            'pnl_vals': self.pnl_vals, 
            'cap_vals': self.cap_vals,
            'all_spreads': self.all_spreads
        }
    
    def compute_bt_metrics(self):
        '''
        Function to screen for quality cointegrated pairs 
        
        Returns
        -------
        dictionary of quality cointegrated pairs with corresponding curated backtesting metrics 
        '''
        
        pnl_pcts, win_pcts, tot_trades, max_wins, max_losses, sharpe_ratios = {}, {}, {}, {}, {}, {}
        num_days = len(self.data.get('dates'))
        
        for key_pair, coint_res in self.coint_pairs.items():
            pnl = self.pnl_vals.get(key_pair)
            tot_pnl = sum(pnl)
            pnl_pct = (self.cap_vals.get(key_pair)[-1] - self.init_cap)/ self.init_cap
            win_trade = sum(1 for i in pnl if i > 0)
            lose_trade = sum(1 for i in pnl if i <= 0)
            tot_trade = win_trade + lose_trade
            win_pct = win_trade/ tot_trade
                  
            if pnl_pct < 0.3 or tot_trade <= num_days//100 or win_pct < 0.3 or coint_res[0] <= 0:
                continue
            else:
                pnl_pct_lst = pd.Series(self.cap_vals.get(key_pair)).pct_change()[1:]
                sharpe_ratio = statistics.mean(pnl_pct_lst)/ statistics.stdev(pnl_pct_lst)
                if sharpe_ratio > 0.5:
                    max_win = max(list(i for i in pnl if i > 0), default=0)
                    max_loss = min(list(i for i in pnl if i <= 0), default=0)
                    
                    pnl_pcts[key_pair] = pnl_pct
                    win_pcts[key_pair] = win_pct
                    tot_trades[key_pair] = tot_trade
                    max_wins[key_pair] = max_win
                    max_losses[key_pair] = max_loss
                    sharpe_ratios[key_pair] = sharpe_ratio
                             
        self.pnl_pcts = pnl_pcts
        self.win_pcts = win_pcts
        self.tot_trades = tot_trades
        self.max_wins = max_wins
        self.max_losses = max_losses
        self.sharpe_ratios = sharpe_ratios
                    
        return {
            'pnl_pcts': self.pnl_pcts,
            'win_pcts': self.win_pcts,
            'tot_trades': self.tot_trades, 
            'max_wins': self.max_wins,
            'max_losses': self.max_losses,
            'sharpe_ratios': self.sharpe_ratios
        }
            
