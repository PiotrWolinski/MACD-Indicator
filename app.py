import pandas as pd

URL = 'https://stooq.pl/q/d/l/?s=wig20&i=d'

df = pd.read_csv('data.csv')

N = len(df.index)

print(df.to_string())
print(N)