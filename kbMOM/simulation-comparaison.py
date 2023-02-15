from crosskbmom import *
from kbmom import KbMOM
from math import floor

from sklearn.datasets import make_blobs
from kbmom.utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def bloc_size(prop_outliers, cste):
    return floor(log(cste) / log(1 - prop_outliers))


def bloc_nb(R, prop_outliers, cste, nb=None):
    if nb is None:
        nb = bloc_size(prop_outliers, cste)
    D = cste - 1 / 2  # (1-prop_outliers)**nb - cste # #
    return floor(log(10 / R) / (2 * D ** 2))


class Simulation():
    MU = np.array([[1, 4], [2, 1], [-2, 3]])

    # parameters for the simulation
    n_samples = 1200
    nb_outliers = 10
    outlier_degree = 20
    repetitions = 10

    all_rmse_c = []
    all_rmse_u = []

    # Main loop on the sub sample size
    for coef_ech in np.arange(30, 130, 10):
        print(coef_ech, end='*')
        # Repetitions
        rmse_c = []
        rmse_u = []
        for j in range(repetitions):
            # data simulation
            X, y_true = make_blobs(n_samples=n_samples, centers=MU, cluster_std=0.4)
            for i in range(nb_outliers):
                X[i, :] = outlier_degree * X[i, :]

            # cross KMOM
            kmom_cross = CrossKbMOM(K=len(MU), coef_ech=coef_ech, nbr_blocks=50)
            kmom_cross.fit(X)
            y_kmom_cross = kmom_cross.predict(X)

            map_kmom_cross = mapping(y_true[nb_outliers:], y_kmom_cross[nb_outliers:])
            rmse_c.append(RMSE(MU, kmom_cross.centers, map_kmom_cross))

            # KbMOM
            kmom_u = KbMOM(K=len(MU), coef_ech=coef_ech, nbr_blocks=50)
            kmom_u.fit(X)
            y_kmom = kmom_u.predict(X)

            map_kmom = mapping(y_true[nb_outliers:], y_kmom[nb_outliers:])
            rmse_u.append(RMSE(MU, kmom_u.centers, map_kmom))

        all_rmse_c.append(rmse_c)
        all_rmse_u.append(rmse_u)

    print('RMSE')
    res = np.array([[np.round(np.median(x), 2) for x in all_rmse_c],
                    [np.round(np.std(x), 2) for x in all_rmse_c],
                    [np.round(np.median(x), 2) for x in all_rmse_u],
                    [np.round(np.std(x), 2) for x in all_rmse_u]
                    ])
    df = pd.DataFrame(res, index=['crossKbMOM med', 'crossKbMOM std', 'KbMOM med', 'KbMOM std'],
                      columns=[str(i) for i in np.arange(30,130,10)])
    df.to_csv('output.csv')
    print(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)
    ax1.violinplot(all_rmse_c, showmedians=True)
    ax1.set_xlabel('cross KbMOM')
    ax1.set_ylabel('rmse')
    ax2.violinplot(all_rmse_u, showmedians=True)
    ax2.set_xlabel('KbMOM')
    fig.suptitle('Influence of the sample size among robust approaches')

    for ax in ax1, ax2:
        ax.grid(True)
        ax.label_outer()

    plt.show()

    @classmethod
    def main(cls):
        pass


if __name__ == '__main__':
    Simulation.main()
