import pandas as pd
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.read_csv('Iris.csv', header=None)
    x = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values
    sc = StandardScaler()
    sc = sc.fit(x)
    print(sc.mean_)
    print(sc.scale_)
    x_train = sc.transform(x)
    print(x_train)