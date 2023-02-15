from crosskbmom import *
from math import log, floor, ceil, sqrt

from sklearn.datasets import make_blobs
from kbmom.utils import *
import matplotlib.pyplot as plt


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

    all_rmse = []

    # Main loop on the sub sample size
    for coef_ech in np.arange(30, 130, 10):
        print(coef_ech, end='*')
        # Repetitions
        rmse = []
        for i in range(repetitions):
            # data simulation
            X, y_true = make_blobs(n_samples=n_samples, centers=MU, cluster_std=0.4)
            for i in range(nb_outliers):
                X[i, :] = outlier_degree * X[i, :]

            kmom_cross = CrossKbMOM(K=len(MU), coef_ech=coef_ech, nbr_blocks=50)
            kmom_cross.fit(X)
            y_kmom = kmom_cross.predict(X)

            map_kmom = mapping(y_true[nb_outliers:], y_kmom[nb_outliers:])
            rmse.append(RMSE(MU, kmom_cross.centers, map_kmom))

        all_rmse.append(rmse)
    print('RMSE')
    for name, x in zip(np.arange(20, 150, 10), all_rmse):
        print('sample size ', name, 'rmse', np.round(np.mean(x), 2))
    #print('centroids')
    #print(kmom_cross.centers.astype(np.int))

    plt.violinplot(all_rmse)
    plt.title('Influence of the sample size of the block')
    plt.show()

    @classmethod
    def main(cls):
        pass


if __name__ == '__main__':
    Simulation.main()
