# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 21:25:25 2021

@author: USUARIO
"""

### Importar Librerias ####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from stocktrends import indicators

#### Importar la data ####

type_bd = int(input('Ingrese el tipo de activo: 1-Indices, 2-Commodities, 3-Criptos: '))

if type_bd == 1:
    data_dict = {1:'Dow_d.csv', 2:'EWG_d.csv', 3:'EWJ_d.csv', 4:'EWP_d.csv', 5:'EWZ_d.csv', 6:'GXC_d.csv', 7:'SPEU_d.csv',
                 8:'SPY_d.csv', 9:'XLE_d.csv', 10:'XLF_d.csv', 11:'XLK_d.csv'}
    activo_dict = {1:'DOW', 2:'EWG', 3:'EWJ', 4:'EWP', 5:'EWZ', 6:'GXC', 7:'SPEU', 8:'SPY', 9:'XLE', 10:'XLF', 11:'XLK'}
    bd = int(input('Eliga la base de datos: 1-DOW, 2-EWG, 3-EWJ, 4-EWP, 5-EWZ, 6-GXC, 7-SPEU, 8-SPY, 9-XLE, 10-XLF, 11-XLK: '))
elif type_bd == 2:
    data_dict = {1:'CO1_d.csv', 2:'GLD_d.csv', 3:'KC1_d.csv', 4:'NG1_d.csv', 5:'BNO_d.csv', 6:'USO_d.csv'}
    activo_dict = {1:'CO', 2:'GLD', 3:'KC1', 4:'NG1', 5:'BNO', 6:'USO'}
    bd = int(input('Eliga la base de datos: 1-CO, 2-GLD, 3-KC1, 4-NG1, 5-BNO, 6-USO: '))
elif type_bd == 3:
    data_dict = {1:'Binance_BTCUSDT_1h.csv', 2:'Binance_ETHUSDT_1h.csv', 3:'Binance_LTCUSDT_1h.csv', 4:'Binance_DASHUSDT_1h.csv',
                 5:'Binance_XLMUSDT_1h.csv', 6:'Binance_ADAUSDT_1h.csv'}
    activo_dict = {1:'BTC', 2:'ETH', 3:'LTC', 4:'DASH', 5:'XLM', 6:'ADA'}
    bd = int(input('Eliga la base de datos: 1-BTC, 2-ETH, 3-LTC, 4-DASH, 5-XLM, 6-ADA: '))

df = pd.read_csv(data_dict[bd])
df = df.sort_index(ascending=False)
df = df.reset_index(drop=True)

df.columns = [i.lower() for i in df.columns]

#### Código Line Break ####

class Instrument:

    def __init__(self, df):
        self.odf = df
        self.df = df
        self._validate_df()

    ohlc = {'open', 'high', 'low', 'close'}

    UPTREND_CONTINUAL = 0
    UPTREND_REVERSAL = 1
    DOWNTREND_CONTINUAL = 2
    DOWNTREND_REVERSAL = 3

    def _validate_df(self):
        if not self.ohlc.issubset(self.df.columns):
            raise ValueError('DataFrame should have OHLC {} columns'.format(self.ohlc))

class LineBreak(Instrument):

    line_number = 3

    def uptrend_reversal(self, close):
        lows = [self.cdf.iloc[i]['low'] for i in range(-1, -self.line_number - 1, -1)]
        least = min(lows)
        return close < least

    def downtrend_reversal(self, close):
        highs = [self.cdf.iloc[i]['high'] for i in range(-1, -self.line_number - 1, -1)]
        highest = max(highs)
        return close > highest

    def get_ohlc_data(self):
        columns = ['date', 'open', 'high', 'low', 'close']
        self.df = self.df[columns]

        self.cdf = pd.DataFrame(columns=columns, data=[])

        for i in range(self.line_number):
            self.cdf.loc[i] = self.df.loc[i]

        self.cdf['uptrend'] = True

        columns = ['date', 'open', 'high', 'low', 'close', 'uptrend']

        for index, row in self.df.iterrows():

            close = row['close']

            row_p1 = self.cdf.iloc[-1]

            uptrend = row_p1['uptrend']

            open_p1 = row_p1['open']
            close_p1 = row_p1['close']

            if uptrend and close > close_p1:
                r = [close_p1, close, close_p1, close]
            elif uptrend and self.uptrend_reversal(close):
                uptrend = not uptrend
                r = [open_p1, open_p1, close, close]
            elif not uptrend and close < close_p1:
                r = [close_p1, close_p1, close, close]
            elif not uptrend and self.downtrend_reversal(close):
                uptrend = not uptrend
                r = [open_p1, close, open_p1, close]
            else:
                continue

            sdf = pd.DataFrame(data=[[row['date']] + r + [uptrend]], columns=columns)
            self.cdf = pd.concat([self.cdf, sdf])

        self.cdf.reset_index(inplace=True)
        return self.cdf

#### Ingresar Parametros Iniciales ####

inferior = int(input('Ingrese la cantidad de lineas inferior: '))
superior = int(input('Ingrese la cantidad de lineas superior: '))
paso = int(input('Ingrese el paso diferencial: '))
capital = float(input('Ingrese la cantidad en dolares a invertir: '))

activo_resumen = (activo_dict[bd] + ' - Line Break ' +' - Capital: ' + str(capital))

#### Correr codigo ####

final_cap = []
total_profits = []
rentabilidad = []

quantity_longs = []
quantity_shorts = []

positive_trades = []
negative_trades = []

positive_long_trades = []
positive_short_trades = []

percentage_long_trades = []
percentage_short_trades = []
percentage_total_trades = []


line_breaks = np.arange(inferior, superior+paso, paso)
'''
for tam_ladri in enumerate(range(inferior, superior+1, paso_int)):
    if tam_ladri[0]==inferior:
        ladrillos.append(tam_ladri[1])
    else:
        ladrillos.append(tam_ladri[1]+paso_float*tam_ladri[0])
    
    tamaño_ladrillo = ladrillos[-1]
'''

for cantidad_lineas in line_breaks:
        
    linebreak = LineBreak(df)
    linebreak.line_number = cantidad_lineas
    data = linebreak.get_ohlc_data()


#### Creacion de tabla de operaciones ####
    
    def get_operations(data):
        date_in = []
        date_out = []
        type_op = []
        price_in = []
        price_out = []
        
        flag = 0
        
        for i in range(1, len(data)):
            if i == len(data)-1:
                date_out.append(data['date'][i])
                price_out.append(data['close'][i])
            else:
                if flag == 0:
                    if data['uptrend'][i] == True:
                        date_in.append(data['date'][i])
                        type_op.append('Long')
                        price_in.append(data['close'][i])
                        flag = 1
                                          
                    elif data['uptrend'][i] == False:
                        date_in.append(data['date'][i])
                        type_op.append('Short')
                        price_in.append(data['close'][i])       
                        flag = -1                                
                                    
                elif flag == 1:
                    if data['uptrend'][i] == False:
                        date_out.append(data['date'][i])
                        price_out.append(data['close'][i])
                        type_op.append('Short')
                        price_in.append(data['close'][i])
                        date_in.append(data['date'][i])
                        flag = -1
                                               
                elif flag == -1:
                    if data['uptrend'][i] == True:
                        date_out.append(data['date'][i])           
                        price_out.append(data['close'][i])
                        type_op.append('Long')
                        price_in.append(data['close'][i])
                        date_in.append(data['date'][i])
                        flag = 1
            
                else:
                    continue
            
        return (date_in, date_out, type_op, price_in, price_out)
    
    data_op = pd.DataFrame()
    
    data_op['Date In'] = get_operations(data)[0]
    data_op['Date Out'] = get_operations(data)[1]
    data_op['Type'] = get_operations(data)[2]
    data_op['Price In'] = get_operations(data)[3]
    data_op['Price Out'] = get_operations(data)[4]
    
    #### Rentabilidad ####
    profits = []
    capital_final = capital
    capital_list = [capital]

    for i in range(len(data_op)):
        if data_op['Type'][i] == 'Short':
            profits.append(capital_final*(data_op['Price In'][i] - data_op['Price Out'][i])/data_op['Price In'][i])
            capital_final += profits[i]
            capital_list.append(capital_final)
            
        elif data_op['Type'][i] == 'Long':
            profits.append(capital_final*(data_op['Price Out'][i] - data_op['Price In'][i])/data_op['Price In'][i])
            capital_final += profits[i]
            capital_list.append(capital_final)
        
    data_op['Profits'] = profits
    
    #### Matriz de Resultados ####
    
    total_profits.append(sum(profits))
    final_cap.append(capital_final)
    rentabilidad.append((capital_final/capital - 1)*100)
    
    quantity_longs.append(len(data_op[data_op.Type == 'Long']))
    quantity_shorts.append(len(data_op[data_op.Type == 'Short']))
    
    positive_trades.append(len(data_op[data_op.Profits > 0]))
    negative_trades.append(len(data_op[data_op.Profits < 0]))
    
    positive_long_trades.append(len(data_op[(data_op.Type == 'Long') & (data_op.Profits > 0)]))
    positive_short_trades.append(len(data_op[(data_op.Type == 'Short') & (data_op.Profits > 0)]))
    
    percentage_long_trades.append(round(positive_long_trades[-1] / quantity_longs[-1] * 100, 2))
    percentage_short_trades.append(round(positive_short_trades[-1] / quantity_shorts[-1] * 100, 2))
    percentage_total_trades.append(round(positive_trades[-1] / (quantity_longs[-1] + quantity_shorts[-1]) * 100, 2))
     
matriz_resultados = pd.DataFrame()
matriz_resultados["Cantidad Lineas"] = line_breaks
matriz_resultados["Total Ganancias"] = total_profits
matriz_resultados['Capital Final'] = final_cap
matriz_resultados['Rentablidad(%)'] = rentabilidad
matriz_resultados["% Trades(+)"] = percentage_total_trades
matriz_resultados["Cant Longs"] = quantity_longs
matriz_resultados["% Longs(+)"] = percentage_long_trades
matriz_resultados["Cant Shorts"] = quantity_shorts
matriz_resultados["% Shorts(+)"] = percentage_short_trades
matriz_resultados["Trades Positivos"] = positive_trades
matriz_resultados["Trades Negativos"] = negative_trades


#### Grafica Matriz de Resultados ####

fig = plt.figure(figsize=(16,8))
plt.bar(matriz_resultados['Cantidad Lineas'], matriz_resultados['Capital Final'])
plt.title(activo_resumen)
#plt.xlim((min(matriz_resultados['Cantidad Lineas']), max(matriz_resultados['Cantidad Lineas'])))
plt.axhline(y=capital, color="black", linewidth=2)
plt.xlabel('Cantidad Lineas')
plt.ylabel('Total Ganancias')
plt.grid()
plt.show()
