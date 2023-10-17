"""
Toy implementation for Improved Deep Embedded Clustering as described in paper:

        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.

The Autoencoder is pretrained directly in an end-to-end manner, NOT greedy layer-wise training. So the results are
different with what reported in the paper.

Usage:
    No pretrained autoencoder weights available:
        python IDEC.py mnist
        python IDEC.py usps
        python IDEC.py reutersidf10k --n_clusters 4
    Weights of Pretrained autoencoder for mnist are in './ae_weights/mnist_ae_weights.h5':
        python IDEC.py mnist --ae_weights ./ae_weights/mnist_ae_weights.h5

Author:
    Xifeng Guo. 2017.4.30
"""

from time import time
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from tensorflow.keras.utils import plot_model

from sklearn.cluster import KMeans
from sklearn import metrics

from DEC import cluster_acc, ClusteringLayer, autoencoder


class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0):

        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder = autoencoder(self.dims)       # создаем автоэнкодер
        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)   #только энкодер от автоэнкодера
        # print('\n11111\n')
        # prepare IDEC model
        # подвешиваем ClusteringLayer с n_clusters нейронами после скрытого слоя (слой будет определять 
        # для представления его сходство с каждым центром)
        # для вычисления выхода ClusteringLayer переопределен метод call в котором qij вычисляется как t-распределение Стьюдента 
        clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        print('clustering_layer =', clustering_layer)
        
        # модель - объединение 2 моделей - автоэнкодера и ClusteringLayer
        # 2 выхода - каждый для своей модели
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])

        self.pretrained = False
        self.centers = []
        self.y_pred = []

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam'):
        print('...Pretraining...')
        self.autoencoder.compile(loss='mse', optimizer=optimizer)  # SGD(lr=0.01, momentum=0.9),
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        self.autoencoder.save_weights('ae_weights.h5')
        print('Pretrained weights are saved to ./ae_weights.h5')
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T   #TODO ошибка? по формле в статье должно быть weight а не weight.T

    # Дивергенция KL и MSE
    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None, save_dir='./results/idec'):

        print('Update interval', update_interval)
        #save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        save_interval = 10e10
        print('Save interval', save_interval)

        # print("\n pretrained 1\n")  
        
        # Step 1: pretrain
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))    #кластеризуем предстваления
        y_pred_last = np.copy(self.y_pred)
        
        #веса слоя ClusteringLayer усатанавливаем равными центрам кластеров представлений
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])   

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/idec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)  # q = выход ClusteringLayer 
                                                         # (то есть сходство с каждым центром кластера для каждого представления), 
                                                         # выход автоэнкодера нас не интересует

                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)       # кластеры представлений
                if y is not None:
                    acc = np.round(cluster_acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logwriter.writerow(dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2]))
                    print('Iter-%d: ACC= %.4f, NMI= %.4f, ARI= %.4f;  L= %.5f, Lc= %.5f,  Lr= %.5f'
                          % (ite, acc, nmi, ari, loss[0], loss[1], loss[2]))

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                
                # критерий останова: tol - доля объектов у которых за одну итерацию изменяются метки кластеров
                # с прошлой итерации меньше которой обучение останавливается
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # print("\n train_on_batch 1\n")        
                    
            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                # Автоэнкодер учим предсказывать собственный выход
                # ClusteringLayer учим предсказывать pij, то есть чтобы qij было как можно ближе к pij
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],    
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

           # print("\n train_on_batch 2\n")      
                
            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                print('saving model to: ' + save_dir + '/IDEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to: ' + save_dir + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')
        
        return self.y_pred

