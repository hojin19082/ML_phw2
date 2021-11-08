import pandas as pd
import numpy as np
import random

import sklearn
import sklearn.preprocessing as pp

import seaborn as sns

import matplotlib.pyplot as plt
#!pip3 install umap-learn
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE
all_data = pd.read_csv('housing.csv')
import warnings
warnings.simplefilter('ignore')
import sys
import time

sns.set_style('whitegrid')

# CLARANS
def euclidean_distance_square(a, b):
    """!
    @brief Calculate square Euclidian distance between vector a and b.
@param[in] a (list): The first vector.
    @param[in] b (list): The second vector.
    @return (double) Square Euclidian distance between two vectors.
    """
    if ( ((type(a) == float) and (type(b) == float)) or ((type(a) == int) and (type(b) == int)) ):
         return (a - b)**2.0

    distance = 0.0
    for i in range(0, len(a)):
        distance += (a[i] - b[i])**2.0;
    return distance

class CLARANS:
    # Source Code From : https://pyclustering.github.io/docs/0.8.2/html/de/d9f/clarans_8py_source.html
    # Added max_iter parameter for stop while loop due to limited time.
    # Added random_state parameter for reproduction
    """!
    @brief Cluster analysis algorithm: CLARANS.
    @details Implementation based on paper @cite article::clarans::1.
    @authors Andrei Novikov (pyclustering@yandex.ru)
    @date 2014-2018
    @copyright GNU Public License
    @cond GNU_PUBLIC_LICENSE
    PyClustering is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    PyClustering is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    @endcond
    """
    def __init__(self, data, number_clusters, numlocal, maxneighbor, max_iter=100, verbose=True, random_state=None):
        self.__pointer_data = data;
        self.__numlocal = numlocal;
        self.__maxneighbor = maxneighbor;
        self.__number_clusters = number_clusters;

        self.__clusters = [];
        self.__current = [];
        self.__belong = [];

        self.__optimal_medoids = [];
        self.__optimal_estimation = float('inf');

        self.max_iter = max_iter
        self.iter = 0
        self.verbose = verbose
        self.seed = random_state

    def process(self):
        random.seed(self.seed);

        for _ in range(0, self.__numlocal):
            # set (current) random medoids
            self.__current = random.sample(range(0, len(self.__pointer_data)), self.__number_clusters);

            # update clusters in line with random allocated medoids
            self.__update_clusters(self.__current);

            # optimize configuration
            self.__optimize_configuration();

            # obtain cost of current cluster configuration and compare it with the best obtained
            estimation = self.__calculate_estimation();
            if (estimation < self.__optimal_estimation):
                self.__optimal_medoids = self.__current[:];
            self.__optimal_estimation = estimation;

        self.__update_clusters(self.__optimal_medoids);
        self.iter = 0

    def get_clusters(self):
        return self.__clusters;

    def get_medoids(self):
        return self.__optimal_medoids;

    def get_cluster_encoding(self):
        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION;

    def __update_clusters(self, medoids):
        self.__belong = [0] * len(self.__pointer_data);
        self.__clusters = [[] for i in range(len(medoids))];
        for index_point in range(len(self.__pointer_data)):
            index_optim = -1;
            dist_optim = 0.0;

            for index in range(len(medoids)):
                dist = euclidean_distance_square(self.__pointer_data[index_point], self.__pointer_data[medoids[index]]);

                if ((dist < dist_optim) or (index is 0)):
                    index_optim = index;
                    dist_optim = dist;

            self.__clusters[index_optim].append(index_point);
            self.__belong[index_point] = index_optim;

        # If cluster is not able to capture object it should be removed
        self.__clusters = [cluster for cluster in self.__clusters if len(cluster) > 0];

    def __optimize_configuration(self):
        index_neighbor = 0;

        while (index_neighbor < self.__maxneighbor) and (self.iter < self.max_iter):
            # get random current medoid that is to be replaced
            current_medoid_index = self.__current[random.randint(0, self.__number_clusters - 1)];
            current_medoid_cluster_index = self.__belong[current_medoid_index];

            # get new candidate to be medoid
            candidate_medoid_index = random.randint(0, len(self.__pointer_data) - 1);

            while (candidate_medoid_index in self.__current):
                candidate_medoid_index = random.randint(0, len(self.__pointer_data) - 1);

            candidate_cost = 0.0;
            for point_index in range(0, len(self.__pointer_data)):
                if (point_index not in self.__current):
                    # get non-medoid point and its medoid
                    point_cluster_index = self.__belong[point_index];
                    point_medoid_index = self.__current[point_cluster_index];

                    # get other medoid that is nearest to the point (except current and candidate)
                    other_medoid_index = self.__find_another_nearest_medoid(point_index, current_medoid_index);
                    other_medoid_cluster_index = self.__belong[other_medoid_index];

                    # for optimization calculate all required distances
                    # from the point to current medoid
                    distance_current = euclidean_distance_square(self.__pointer_data[point_index],
                                                                 self.__pointer_data[current_medoid_index]);

                    # from the point to candidate median
                    distance_candidate = euclidean_distance_square(self.__pointer_data[point_index],
                                                                   self.__pointer_data[candidate_medoid_index]);

                    # from the point to nearest (own) medoid
                    distance_nearest = float('inf');
                    if ((point_medoid_index != candidate_medoid_index) and (
                            point_medoid_index != current_medoid_cluster_index)):
                        distance_nearest = euclidean_distance_square(self.__pointer_data[point_index],
                                                                     self.__pointer_data[point_medoid_index]);

                    # apply rules for cost calculation
                    if (point_cluster_index == current_medoid_cluster_index):
                        # case 1:
                        if (distance_candidate >= distance_nearest):
                            candidate_cost += distance_nearest - distance_current;

                        # case 2:
                        else:
                            candidate_cost += distance_candidate - distance_current;

                    elif (point_cluster_index == other_medoid_cluster_index):
                        # case 3 ('nearest medoid' is the representative object of that cluster and object is more similar to 'nearest' than to 'candidate'):
                        if (distance_candidate > distance_nearest):
                            pass;

                        # case 4:
                        else:
                            candidate_cost += distance_candidate - distance_nearest;

            if (candidate_cost < 0):
                # set candidate that has won
                self.__current[current_medoid_cluster_index] = candidate_medoid_index;

                # recalculate clusters
                self.__update_clusters(self.__current);

                # reset iterations and starts investigation from the begining
                index_neighbor = 0;

            else:
                index_neighbor += 1;

            self.iter += 1
            if self.verbose:
                sys.stdout.write(f"Iteration : {self.iter}\r")
                if self.iter >= self.max_iter:
                    print(f"Max iter {self.max_iter} Reached. Returning Current Best Result.")

    def __find_another_nearest_medoid(self, point_index, current_medoid_index):
        other_medoid_index = -1;
        other_distance_nearest = float('inf');
        for index_medoid in self.__current:
            if (index_medoid != current_medoid_index):
                other_distance_candidate = euclidean_distance_square(self.__pointer_data[point_index],
                                                                     self.__pointer_data[current_medoid_index]);

                if (other_distance_candidate < other_distance_nearest):
                    other_distance_nearest = other_distance_candidate;
                    other_medoid_index = index_medoid;

        return other_medoid_index;

    def __calculate_estimation(self):
        estimation = 0.0;
        for index_cluster in range(0, len(self.__clusters)):
            cluster = self.__clusters[index_cluster];
            index_medoid = self.__current[index_cluster];
            for index_point in cluster:
                estimation += euclidean_distance_square(self.__pointer_data[index_point],
                                                        self.__pointer_data[index_medoid]);

        return estimation;


# Helper functions
def random_feature_selection(data, num_features):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    feature_names = list(data)
    selected_features = random.sample(feature_names, num_features)

    return data[selected_features]


def df_fillna(data, fill, cols=None):
    x = data.copy()

    if cols is not None:
        feature_names = cols
    else:
        feature_names = list(data)

        for col in feature_names:
            if x[col].isnull().sum() == 0:
                pass
            elif fill == 'mean':
                x[col].fillna(x[col].mean(), inplace=True)
            elif fill == 'median':
                x[col].fillna(x[col].median(), inplace=True)
            elif fill == 'mode':
                x[col].fillna(x[col].mode()[0], inplace=True)
            else:
                x[col].fillna(fill, inplace=True)
        return x

def encode_and_scale(data, encoder, scaler, cols='auto'):
    if cols is None:
        feature_names = list(data)
    elif cols == 'auto':
        feature_names = list(data.select_dtypes(object))
    else:
        feature_names = cols

    x = data.copy()
    if len(feature_names) != 0:
        encoded = pd.DataFrame(encoder.fit_transform(x[feature_names]))
        x.drop(columns=feature_names, inplace=True)
        x = pd.concat((x, encoded), axis=1)

    x = scaler.fit_transform(x)
    return x

def run_cfg(cfg, x, verbose=False, random_state=18, score_only=False):
    data_selected = x[cfg['selected_features']]
    data_selected = df_fillna(data_selected, fill=cfg['fill'], cols=None)
    data_selected = encode_and_scale(data_selected,
                                         encoder=cfg['encoder'],
                                         scaler=cfg['scaler'])

    kmeans = KMeans(**cfg['kmeans_params'])
    kmeans_pred = kmeans.fit_predict(data_selected)
    kmeans_cluster = pd.DataFrame(kmeans_pred)

    try:
        kmeans_score = sklearn.metrics.silhouette_score(data_selected, kmeans_pred,
                                                        metric='l2', random_state=random_state)
    except ValueError:
        kmeans_score = -1
    if verbose:
        print(f"KMeans : {kmeans_score}")

    gmm = GaussianMixture(**cfg['gmm_params'])
    gmm_pred = gmm.fit_predict(data_selected)
    gmm_cluster = pd.DataFrame(gmm_pred)
    try:
        gmm_score = sklearn.metrics.silhouette_score(data_selected, gmm_pred,metric='l2', random_state=random_state)
    except ValueError:
        gmm_score = -1
    if verbose:
        print(f"GMM : {gmm_score}")

    dbscan = DBSCAN(**cfg['dbscan_params'])
    dbscan_pred = dbscan.fit_predict(data_selected)
    dbscan_cluster = pd.DataFrame(dbscan_pred)
    try:
        dbscan_score = sklearn.metrics.silhouette_score(data_selected, dbscan_pred,
                                                   metric='l2', random_state=random_state)
    except ValueError:
        dbscan_score = -1
    if verbose:
        print(f"DBSCAN : {dbscan_score}")


    clarans = CLARANS(data_selected.tolist(), **cfg['clarans_params'])
    clarans.process()
    clarans_pred = -np.ones(shape=data_selected.shape[0])
    clarans_cluster = pd.DataFrame(clarans_pred)
    for i, idx in enumerate(clarans.get_clusters()):
        clarans_pred[idx] = i
    try:
        clarans_score = sklearn.metrics.silhouette_score(data_selected, clarans_pred,
                                                         metric='l2', random_state=random_state)
    except ValueError:
        clarans_score = -1
    if verbose:
        print(f"CLARANS : {clarans_score}")

    meanshift = MeanShift(**cfg['meanshift_params'])
    meanshift_pred = meanshift.fit_predict(data_selected)
    meanshift_cluster = pd.DataFrame(meanshift_pred)
    try:
        meanshift_score = sklearn.metrics.silhouette_score(data_selected, meanshift_pred,
                                                      metric='l2', random_state=random_state)
    except ValueError:
        meanshift_score = -1
    if verbose:
        print(f"Meanshift : {meanshift_score}")

    if score_only:
        return np.mean([kmeans_score, gmm_score, dbscan_score, clarans_score, meanshift_score])

    # Cluster labels for each clustering model
    kmeans_cluster_size = kmeans_cluster.value_counts()
    gmm_cluster_size = gmm_cluster.value_counts()
    dbscan_cluster_size = dbscan_cluster.value_counts()
    clarans_cluster_size = clarans_cluster.value_counts()
    meanshift_cluster_size = meanshift_cluster.value_counts()

    # Barplot to visualize the number of data in each cluster.
    figure, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5)
    figure.set_size_inches(15, 5)

    kmeans_cluster_size.plot(kind='bar', ax=ax1)
    ax1.set_title('Kmeans')
    gmm_cluster_size.plot(kind='bar', ax=ax2)
    ax2.set_title('GMM')
    dbscan_cluster_size.plot(kind='bar', ax=ax3)
    ax3.set_title('DBSCAN')
    clarans_cluster_size.plot(kind='bar', ax=ax4)
    ax4.set_title('Clarans')
    meanshift_cluster_size.plot(kind='bar', ax=ax5)
    ax5.set_title('Meanshift')
    plt.show()

    tsne = TSNE(perplexity=30, early_exaggeration=12, learning_rate=100,
                n_iter=1000, n_iter_without_progress=200, verbose=verbose)
    data_embedded = tsne.fit_transform(data_selected)

    figure, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1)
    figure.set_size_inches(10, 10 * 5)
    sns.scatterplot(x=data_embedded[:, 0], y=data_embedded[:, 1], hue=kmeans_pred, ax=ax1)
    sns.scatterplot(x=data_embedded[:, 0], y=data_embedded[:, 1], hue=gmm_pred, ax=ax2)
    sns.scatterplot(x=data_embedded[:, 0], y=data_embedded[:, 1], hue=dbscan_pred, ax=ax3)
    sns.scatterplot(x=data_embedded[:, 0], y=data_embedded[:, 1], hue=clarans_pred, ax=ax4)
    sns.scatterplot(x=data_embedded[:, 0], y=data_embedded[:, 1], hue=meanshift_pred, ax=ax5)

    score_list = [kmeans_score, gmm_score, dbscan_score, clarans_score, meanshift_score]
    clusters = np.stack((kmeans_pred, gmm_pred, dbscan_pred, clarans_pred, meanshift_pred), axis=1)

    return score_list, clusters, data_embedded, data_selected


def random_search_clustering(x, n_trials=100, random_state=18, verbose=False):
    cfgs_list = []

    for i in tqdm(range(n_trials)):
        data = x.copy()
        cfg = {}
        score = 0

        cfg['selected_features'] = random.sample(list(X), random.randint(1, len(data.columns)))
        cfg['fill'] = random.choice(['mean', 'median', 'mode'])
        cfg['encoder'] = random.choice(
            [pp.OneHotEncoder(sparse=False),
             pp.OrdinalEncoder()])

        cfg['scaler'] = random.choice(
            [pp.StandardScaler(),
             pp.RobustScaler(),
             pp.MinMaxScaler(),
             pp.MaxAbsScaler()])
        cfg['n_clusters'] = random.randint(2, 12) # 2 to 12

        # Random Kmeans Parameters
        cfg['kmeans_params'] = {'random_state' : random_state,
                                'n_clusters' : cfg['n_clusters'],
                               }
        # Random EM Parameters
        cfg['gmm_params'] = {'random_state' : random_state,
                             'n_components' : cfg['n_clusters'],
                             'covariance_type' : random.choice(['full','tied','diag','spherical']),
                             'reg_covar' : np.log(random.uniform(1, np.e**5))
                            }

        # Random DBSCAN Parameters
        cfg['dbscan_params'] = {'eps' : random.uniform(0.1, 10),
                                'min_samples' : random.randint(2, 100),
                                'metric' : 'minkowski',
                                'p' : random.randint(1, 9),
                               }

        # Random CLARANS Parameters
        cfg['clarans_params'] = {'number_clusters' : cfg['n_clusters'],
                                'numlocal' : random.randint(1, 10),
                                'maxneighbor' : random.randint(2, 50),
                                'max_iter' : 100,
                                'verbose' : verbose,
                                'random_state' : random.uniform(0, 18181818)
                                }

        # Random Mean Shift Parameters
        cfg['meanshift_params'] = {'bandwidth' : np.log(random.uniform(1, np.e**3)),
                                  'min_bin_freq' : random.randint(1, 30),
                                  'n_jobs' : 2}

        score = run_cfg(cfg, data, verbose=verbose, random_state=18, score_only=True)
        cfg['score'] = score
        if verbose:
            print(f"Score : {cfg['score']}")
        cfgs_list.append(cfg)
        time.sleep(0.5)
    cfgs_list = pd.DataFrame.from_dict(cfgs_list)
    cfgs_list = cfgs_list.sort_values(by='score', ascending=False)

    best_cfg = dict(cfgs_list.iloc[0])
    scores, cluster_predictions, data_embedding, data_selected = run_cfg(best_cfg, data, verbose=verbose, random_state=18, score_only=False)
    return best_cfg, scores, cluster_predictions, data_embedding, data_selected

# Run Pipeline
X = all_data.drop(columns='median_house_value')
Y = all_data[['median_house_value']]
X[X.select_dtypes(np.number).columns] = X[X.select_dtypes(np.number).columns].astype('float32')
best_cfg, scores, cluster_predictions, data_embedding, data_selected  = random_search_clustering(X, n_trials=1, verbose=True, random_state=18)

# Run after random_search_clustering() with single ‘best_cfg’ dict.
# Y : median_house_value feature defined above.

# Interpret Results
def get_quantile_label(label, n_quantiles):
    l = label.copy()
    quantiles = np.quantile(np.array(label), [i / n_quantiles for i in range(n_quantiles)])
    q_group = []
    for val in np.array(label):
        i = 0
        for q in quantiles:
            if val >= q:
                i += 1
        q_group.append(i)
    l['quantile_groups'] = q_group
    return l
Yq = get_quantile_label(Y, n_quantiles=best_cfg['n_clusters']).astype('int')
Yq[['Kmeans','GMM','DBSCAN','CLARANS','MeanShift']] = cluster_predictions.astype('int')

# Distribution Plot
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
figure.set_size_inches(18, 18)
sns.histplot(data=Yq, x='median_house_value', hue='quantile_groups', ax=ax1, kde=True)
sns.histplot(data=Yq, x='median_house_value', hue='Kmeans', ax=ax2, kde=True)
sns.histplot(data=Yq, x='median_house_value', hue='GMM', ax=ax3, kde=True)
sns.histplot(data=Yq, x='median_house_value', hue='DBSCAN', ax=ax4, kde=True)
sns.histplot(data=Yq, x='median_house_value', hue='CLARANS', ax=ax5, kde=True)
sns.histplot(data=Yq, x='median_house_value', hue='MeanShift', ax=ax6, kde=True)

# Data Embedding visualization via tSNE
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
figure.set_size_inches(18, 18)
sns.scatterplot(x=data_embedding[:, 0], y=data_embedding[:, 1], hue=Yq['quantile_groups'], ax=ax1)
sns.scatterplot(x=data_embedding[:, 0], y=data_embedding[:, 1], hue=Yq['Kmeans'], ax=ax2)
sns.scatterplot(x=data_embedding[:, 0], y=data_embedding[:, 1], hue=Yq['GMM'], ax=ax3)
sns.scatterplot(x=data_embedding[:, 0], y=data_embedding[:, 1], hue=Yq['DBSCAN'], ax=ax4)
sns.scatterplot(x=data_embedding[:, 0], y=data_embedding[:, 1], hue=Yq['CLARANS'], ax=ax5)
sns.scatterplot(x=data_embedding[:, 0], y=data_embedding[:, 1], hue=Yq['MeanShift'], ax=ax6)

# Barplot to visualize Specific Cluster
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
figure.set_size_inches(18, 18)
sns.barplot(x=X['ocean_proximity'], y=Yq['median_house_value'], hue=Yq['quantile_groups'], ax=ax1)
sns.barplot(x=X['ocean_proximity'], y=Yq['median_house_value'], hue=Yq['Kmeans'], ax=ax2)
sns.barplot(x=X['ocean_proximity'], y=Yq['median_house_value'], hue=Yq['GMM'], ax=ax3)
sns.barplot(x=X['ocean_proximity'], y=Yq['median_house_value'], hue=Yq['DBSCAN'], ax=ax4)
sns.barplot(x=X['ocean_proximity'], y=Yq['median_house_value'], hue=Yq['CLARANS'], ax=ax5)
sns.barplot(x=X['ocean_proximity'], y=Yq['median_house_value'], hue=Yq['MeanShift'], ax=ax6)
# Countplot of each cluster
figure, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5)
figure.set_size_inches(25, 5)
sns.countplot(Yq['Kmeans'], ax=ax1)
sns.countplot(Yq['GMM'], ax=ax2)
sns.countplot(Yq['DBSCAN'], ax=ax3)
sns.countplot(Yq['CLARANS'], ax=ax4)
sns.countplot(Yq['MeanShift'], ax=ax5)


