# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:30:04 2021

@author: USUARIO
"""

#pip install xlrd
#pip install openpyxl

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
    data_dict = {1:'Binance_BTCUSDT_30m.csv', 2:'Binance_ETHUSDT_1h.csv', 3:'Binance_SOLUSDT_60m.csv', 4:'Binance_XRPUSDT_60m.csv',
                 5:'Binance_DOTUSDT_30m.csv', 6:'Binance_ADAUSDT_2h.csv'}
    activo_dict = {1:'BTC', 2:'ETH', 3:'SOL', 4:'XRP', 5:'DOT', 6:'ADA'}
    bd = int(input('Eliga la base de datos: 1-BTC, 2-ETH, 3-SOL, 4-XRP, 5-DOT, 6-ADA: '))

df = pd.read_csv(data_dict[bd])
df = df.sort_index(ascending=False)
df = df.reset_index(drop=True)

df.columns = [i.lower() for i in df.columns]


#### Codigo Renko ####

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

class Renko(Instrument):

    PERIOD_CLOSE = 1
    PRICE_MOVEMENT = 2

    TREND_CHANGE_DIFF = 2

    brick_size = 1
    chart_type = PERIOD_CLOSE

    def get_ohlc_data(self):
        if self.chart_type == self.PERIOD_CLOSE:
            self.period_close_bricks()
        else:
            self.price_movement_bricks()

        return self.cdf

    def price_movement_bricks(self):
        pass

    def period_close_bricks(self):
        brick_size = self.brick_size
        columns = ['date', 'open', 'high', 'low', 'close']
        self.df = self.df[columns]

        self.cdf = pd.DataFrame(
            columns=columns,
            data=[],
        )

        self.cdf.loc[0] = self.df.loc[0]
        close = self.df.loc[0]['close'] // brick_size * brick_size
        self.cdf.iloc[0, 1:] = [close - brick_size, close, close - brick_size, close]
        self.cdf['uptrend'] = True
        

        columns = ['date', 'open', 'high', 'low', 'close', 'uptrend', 'real_close']

        for index, row in self.df.iterrows():

            close = row['close']
            date = row['date']

            row_p1 = self.cdf.iloc[-1]
            uptrend = row_p1['uptrend']
            close_p1 = row_p1['close']

            bricks = int((close - close_p1) / brick_size)
            data = []

            if uptrend and bricks >= 1:
                for i in range(bricks):
                    r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size, uptrend, close]
                    data.append(r)
                    close_p1 += brick_size
            elif uptrend and bricks <= -2:
                uptrend = not uptrend
                bricks += 1
                close_p1 -= brick_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size, uptrend, close]
                    data.append(r)
                    close_p1 -= brick_size
            elif not uptrend and bricks <= -1:
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size, uptrend, close]
                    data.append(r)
                    close_p1 -= brick_size
            elif not uptrend and bricks >= 2:
                uptrend = not uptrend
                bricks -= 1
                close_p1 += brick_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size, uptrend, close]
                    data.append(r)
                    close_p1 += brick_size
            else:
                continue

            sdf = pd.DataFrame(data=data, columns=columns)
            self.cdf = pd.concat([self.cdf, sdf])

        self.cdf.reset_index(inplace=True, drop=True)
        return self.cdf

    def shift_bricks(self):
        shift = self.df['close'].iloc[-1] - self.bdf['close'].iloc[-1]
        if abs(shift) < self.brick_size:
            return
        step = shift // self.brick_size
        self.bdf[['open', 'close']] += step * self.brick_size

#### Ingresar Parametros Iniciales ####

tamaño_ladrillo = float(input('Ingrese el tamaño del ladrillo: '))
estrategia = int(input('Ingrese la cantidad de ladrillos para confirmación: '))
capital = float(input('Ingrese la cantidad en dolares a invertir: '))

activo_resumen = (activo_dict[bd] + ' - Tamaño Ladrillo:' + str(tamaño_ladrillo) 
                                    + ' - Confirmación:' + str(estrategia) 
                                    +' - Capital Inicial:' + str(capital))

#### Correr codigo ####
renko = Renko(df)

renko.brick_size = tamaño_ladrillo
renko.chart_type = Renko.PERIOD_CLOSE
data = renko.get_ohlc_data()

data['Color'] = np.where(data['uptrend'] == True, 'Green', 'Red')      

#### Creacion de tabla de operaciones ####

def get_operations(data):
    date_in = []
    date_out = []
    type_op = []
    price_in = []
    price_out = []
    
    flag = 0
    counter_true = 0
    counter_false = 0
    
    for i in range(1, len(data)):
        if i == len(data)-1:
            date_out.append(data['date'][i])
            price_out.append(data['real_close'][i])
        else:
            if flag == 0:
                if data['uptrend'][i] == True:
                    counter_true += 1                    
                    if counter_true >= estrategia:
                        date_in.append(data['date'][i])
                        type_op.append('Long')
                        price_in.append(data['real_close'][i])
                        flag = 1
                                      
                elif data['uptrend'][i] == False:
                    counter_false += 1                    
                    if counter_false >= estrategia:
                        date_in.append(data['date'][i])
                        type_op.append('Short')
                        price_in.append(data['real_close'][i])       
                        flag = -1                                
                                
            elif flag == 1:
                if data['uptrend'][i] == False:
                    counter_false += 1                    
                    if counter_false >= estrategia:
                        counter_true = 0
                        date_out.append(data['date'][i])
                        price_out.append(data['real_close'][i])
                        type_op.append('Short')
                        price_in.append(data['real_close'][i])
                        date_in.append(data['date'][i])
                        flag = -1
                else:
                    counter_false = 0
                                           
            elif flag == -1:
                if data['uptrend'][i] == True:
                    counter_true += 1                
                    if counter_true >= estrategia:
                        counter_false = 0
                        date_out.append(data['date'][i])           
                        price_out.append(data['real_close'][i])
                        type_op.append('Long')
                        price_in.append(data['real_close'][i])
                        date_in.append(data['date'][i])
                        flag = 1
                else:
                    counter_true = 0
        
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

#### Grafica de Renko ####
def plot_renko(data):
    
    renkos = zip(data['open'],data['close'])
    
    fig = plt.figure(figsize=(16,8))
    
    fig.clf()
    axes = fig.gca()
    
    index = 0
    for open_price, close_price in renkos:
        if (open_price < close_price):
            renko = Rectangle((index,open_price), 1, close_price-open_price, edgecolor='darkblue', facecolor='blue', alpha=0.5)
            axes.add_patch(renko)
        else:
            renko = Rectangle((index,open_price), 1, close_price-open_price, edgecolor='darkred', facecolor='red', alpha=0.5)
            axes.add_patch(renko)
        index = index + 1
        
    num_bars = len(data)
    
    # adjust the axes
    plt.xlim([0, num_bars])
    plt.ylim([min(min(data['open']),min(data['close'])), max(max(data['open']),max(data['close']))])
    fig.suptitle(activo_resumen)
    #fig.suptitle('Bars from ' + min(data['date_time']).strftime("%d-%b-%Y %H:%M") + " to " + max(data['date_time']).strftime("%d-%b-%Y %H:%M") \
    #    + '\nPrice movement = ' + str(price_move), fontsize=14)
    plt.xlabel('Bar Number')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

plot_renko(data)

#### Matriz de Resultados ####
'''
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
'''
data.to_excel('filename.xlsx')

#Matriz_resultados.to_excel('nombre.xlsx')

#plt.plot(df['close'])

