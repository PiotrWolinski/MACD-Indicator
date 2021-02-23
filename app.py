import pandas as pd
from pandas import DataFrame

URL = 'https://stooq.pl/q/d/l/?s=wig20&i=d'
SET_SIZE = 1000

df = pd.read_csv(URL)

def EMA(df: DataFrame, period: int, offset: int) -> float:
    a = 2 / (period + 1)
    cur_day = len(df.index) - offset
    upper_sum = 0.0
    lower_sum = 0.0

    power = 0
    for i in range(cur_day - 1, cur_day - 1 - period - 1, -1):
        upper_sum += df.loc[i][4] * (1 - a) ** power
        lower_sum += (1 - a) ** power
        power += 1

    return upper_sum / lower_sum

def get_MACD(df: DataFrame) -> list[float]:
    MACD = []

    for i in range(SET_SIZE):
        MACD.append(EMA(df, period=12, offset=i) - EMA(df, period=26, offset=i))

    return MACD



def main():
    pass


if __name__ == '__main__':
    main()

