# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:29:57 2021

@author: USUARIO
"""

#### Importar Librerias ####
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

    line_number = 10

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

cantidad_lineas = int(input('Ingrese la cantidad de lineas para ruptura: '))
capital = float(input('Ingrese la cantidad en dolares a invertir: '))

activo_resumen = (activo_dict[bd] + ' - Cantidad de Lineas:' + str(cantidad_lineas) 
                                  + ' - Capital Inicial:' + str(capital))

#### Correr codigo ####
linebreak = LineBreak(df)
linebreak.line_number = cantidad_lineas
data = linebreak.get_ohlc_data()

data['state'] = np.where(data['uptrend'] == True, 'Long', 'Short')

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

data_op['diff'] = abs(data_op['Price In'] - data_op['Price Out'])
data_op['Profits'] = profits
data_op['Capital'] = capital_list[1:]


#### Grafica LineBreak ####
def plot_linebreak(data):
    
    linebreaks = zip(data['open'],data['close'])
    
    fig = plt.figure(figsize=(16,8))
    
    fig.clf()
    axes = fig.gca()
    
    index = 1
    for open_price, close_price in linebreaks:
        if (open_price < close_price):
            linebreak = Rectangle((index,open_price), 1, close_price-open_price, edgecolor='darkblue', facecolor='blue', alpha=0.5)
            axes.add_patch(linebreak)
        else:
            linebreak = Rectangle((index,open_price), 1, close_price-open_price, edgecolor='darkred', facecolor='red', alpha=0.5)
            axes.add_patch(linebreak)
        index = index + 1
        
    num_bars = len(data)
    
    # adjust the axes
    plt.xlim([0, num_bars])
    plt.ylim([min(min(data['open']),min(data['close'])), max(max(data['open']),max(data['close']))])
    fig.suptitle(activo_resumen)
    plt.xlabel('Bar Number')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

plot_linebreak(data)

#### Matriz de Resultados ####

total_profits = sum(profits)
final_cap = capital_final
rentabilidad = round((capital_final/capital-1)*100, 2)

quantity_longs = len(data_op[data_op.Type == 'Long'])
quantity_shorts = len(data_op[data_op.Type == 'Short'])

positive_trades = len(data_op[data_op.Profits > 0])
negative_trades = len(data_op[data_op.Profits < 0])

positive_long_trades = len(data_op[(data_op.Type == 'Long') & (data_op.Profits > 0)])
positive_short_trades = len(data_op[(data_op.Type == 'Short') & (data_op.Profits > 0)])

percentage_long_trades = round(positive_long_trades / quantity_longs * 100, 2)
percentage_short_trades = round(positive_short_trades / quantity_shorts * 100, 2)
percentage_total_trades = round(positive_trades / (quantity_longs + quantity_shorts) * 100, 2)

Matriz_resultados = pd.DataFrame(data={'Matriz de Resultados':['Total Ganancia', 'Capital Final', 'Rentabilidad(%)', '% Trades +',
                                                               'Cant Longs', '% Longs +',
                                                               'Cant Shorts',  '% Shorts +', 
                                                               'Trades Positivos', 'Trades Negativos'],
                                      'Valor':[total_profits, final_cap, rentabilidad, percentage_total_trades,
                                               quantity_longs, percentage_long_trades,
                                               quantity_shorts, percentage_short_trades, 
                                               positive_trades, negative_trades]})
Matriz_resultados.round({'Valor':2})


comision = len(Matriz_resultados) * 0.5

#### Grafica Acumulado
fig = plt.figure(figsize=(16,8))
plt.plot(capital_list)
plt.title('CAPITAL ' + activo_resumen + '\n Capital Final: ' + str(round(capital_final, 2))
                     + ' Rentabilidad(%): ' + str(rentabilidad))
plt.xlabel('Operaciones')
plt.ylabel('Capital')
plt.axhline(y=0, color="black", linewidth=2)
plt.axhline(y=capital, color="black", linewidth=2)
plt.grid(True)
plt.show()


#### Estado Operación


if data['state'].iloc[-1] != data['state'].iloc[-2]:
    print('Ojo, cambio de estado\nAnterior:', data['state'].iloc[-2], '\nAhora:', data['state'].iloc[-1])
else:
    print('Ultimo estado:', data['state'].iloc[-1])


print(activo_dict[bd], 'estado:', data['state'].iloc[-1])



