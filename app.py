import pandas as pd
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from pandas import DataFrame


URL = 'https://stooq.pl/q/d/l/?s=wig20&i=d'

SET_SIZE = 1000
X_LIMIT = [SET_SIZE // 10, SET_SIZE]
Y_LIMIT = [-200, 200]

# SET_SIZE = 300
# X_LIMIT = [SET_SIZE - 100, SET_SIZE]
# Y_LIMIT = [-200, 200]


def EMA(arr: ndarray, period: int, offset: int) -> float:
    """Returns EMA based on the close value, from last days
    specified by the period variable.
    Offset is counted from the current day.
    """
    a = 2 / (period + 1)
    cur_day = arr.size - offset
    upper_sum = 0.0
    lower_sum = 0.0

    power = 0
    for i in range(cur_day - 1, cur_day - 1 - period - 1, -1):
        upper_sum += arr[i] * (1 - a) ** power
        lower_sum += (1 - a) ** power
        power += 1

    return upper_sum / lower_sum

def get_MACD(df: DataFrame, records: int=SET_SIZE) -> ndarray:
    MACD = []
    values = df.to_numpy()
    values = values[:,4]

    for i in range(records, 0, -1):
        MACD.append(EMA(values, period=12, offset=i) - EMA(values, period=26, offset=i))

    return np.asarray(MACD)

def get_signal(arr: ndarray, records: int=SET_SIZE) -> ndarray:
    signal = []

    for i in range(records, 0, -1):
        signal.append(EMA(arr, period=9, offset=i))

    return np.asarray(signal)

def plot_values(df: DataFrame) -> None:
    x = [i for i in range(SET_SIZE)]
    
    val = df.to_numpy()
    val = val[:, 4]
    
    val = val[val.size - SET_SIZE::]
    
    plt.plot(x, val)
    plt.xlabel('Days')
    plt.ylabel('Stock closing values')
    plt.legend()
    plt.title('WIG20 values')
    axes = plt.gca()
    axes.set_xlim(X_LIMIT)
    plt.show()

def plot_MACD(df: DataFrame) -> None:
    x = [i for i in range(SET_SIZE)]

    MACD = get_MACD(df)
    signal = get_signal(MACD)    

    plt.plot(x, MACD, label='MACD')
    plt.plot(x, signal, label='Signal')
    plt.xlabel('Days')
    plt.ylabel('MACD value')
    plt.legend()
    plt.title('MACD indicator')
    axes = plt.gca()
    axes.set_xlim(X_LIMIT)
    axes.set_ylim(Y_LIMIT)
    plt.show()

def simulate(df: DataFrame):
    resources = 1000

    MACD = get_MACD(df)
    signal = get_signal(MACD)  

    for i in range(1, MACD.size):
        if MACD[i - 1] > signal[i - 1] and MACD[i] < signal[i]:
            # Selling signal
            pass
        elif MACD[i - 1] < signal[i - 1] and MACD[i] > signal[i]:
            # Buying signal
            pass
        else:
            # Do nothing
            pass
    pass
    
def predict_action(df: DataFrame) -> str:
    pass


def main():
    df = pd.read_csv(URL)
    plot_MACD(df)
    plot_values(df)

if __name__ == '__main__':
    main()
