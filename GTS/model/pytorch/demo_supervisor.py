import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from lib import utils
from model.pytorch.model import GTSModel
from model.pytorch.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import pandas as pd
import os
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GTSDemoSupervisor:
    def __init__(self, save_adj_name, temperature, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._other_kwargs = kwargs.get('others')
        self.temperature = float(temperature)
        self.opt = self._train_kwargs.get('optimizer')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.ANNEAL_RATE = 0.00003
        self.temp_min = 0.1
        self.save_adj_name = save_adj_name
        self.epoch_use_regularization = self._train_kwargs.get('epoch_use_regularization')
        self.num_sample = self._train_kwargs.get('num_sample')

        #model directory 
        self.model_dir = self._model_kwargs.get('model_dir')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_demo_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        ### Feas
        if self._data_kwargs['dataset_dir'] == 'data/Nottingham':
            df = pd.read_csv('./data/Nottingham_timestamp_availability_data.csv', index_col='timestamp')
        elif self._data_kwargs['dataset_dir'] == 'data/SFPark':
            df = pd.read_csv('./data/sfpark_timestamp_availability_data.csv', index_col='timestamp')
        #else:
        #    df = pd.read_csv('./data/pmu_normalized.csv', header=None)
        #    df = df.transpose()
        num_samples = df.shape[0]
        num_train = round(num_samples * 0.7)
        df = df[:num_train].values
        scaler = utils.StandardScaler(mean=df.mean(), std=df.std())
        train_feas = scaler.transform(df)
        self._train_feas = torch.Tensor(train_feas).to(device)
        #print(self._train_feas.shape)

        k = self._train_kwargs.get('knn_k')
        knn_metric = 'cosine'
        from sklearn.neighbors import kneighbors_graph
        g = kneighbors_graph(train_feas.T, k, metric=knn_metric)
        g = np.array(g.todense(), dtype=np.float32)
        self.adj_mx = torch.Tensor(g).to(device)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder
        self.index_list = self._other_kwargs.get('index_list')
        self.duration_list = self._other_kwargs.get('duration_list')

        # setup model
        GTS_model = GTSModel(self.temperature, self._logger, **self._model_kwargs)
        self.GTS_model = GTS_model.cuda() if torch.cuda.is_available() else GTS_model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'GTS_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('%s/epo%d.tar' % (self.model_dir, self._epoch_num)), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('%s/epo%d.tar' % (self.model_dir, self._epoch_num), map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']
        self.GTS_model.load_state_dict(model_state_dict)
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.GTS_model = self.GTS_model.eval()

            demo_iterator = self._data['demo_loader'].get_iterator()

            for _, (x, y) in enumerate(demo_iterator):
                x, y = self._prepare_data(x, y)
                output = self.GTS_model(inputs=x, node_feas=self._train_feas, temp=self.temperature)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self,label, dataset='val', batches_seen=0, gumbel_soft=True):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.GTS_model = self.GTS_model.eval()

            demo_iterator = self._data['demo_loader'].get_iterator()
            losses = []
            mapes = []
            #rmses = []
            mses = []
            temp = self.temperature
            
            l_1 = []
            m_1 = []
            r_1 = []
            l_2 = []
            m_2 = []
            r_2 = []
            l_3 = []
            m_3 = []
            r_3 = []
            l_4 = []
            m_4 = []
            r_4 = []

            #indexes 
            index1 = self.index_list[0]
            index2 = self.index_list[1]
            index3 = self.index_list[2]
            index4 = self.index_list[3]

            for batch_idx, (x, y) in enumerate(demo_iterator):
                x, y = self._prepare_data(x, y)

                output, mid_output = self.GTS_model(inputs=x, node_feas=self._train_feas, temp=temp)

                if label == 'without_regularization': 
                    loss = self._compute_loss(y, output)
                    y_true = self.standard_scaler.inverse_transform(y)
                    y_pred = self.standard_scaler.inverse_transform(output)
                    mapes.append(masked_mape_loss(y_pred, y_true).item())
                    mses.append(masked_mse_loss(y_pred, y_true).item())
                    #rmses.append(masked_rmse_loss(y_pred, y_true).item())
                    losses.append(loss.item())
                    
                    
                    # Followed the DCRNN TensorFlow Implementation
                    l_1.append(masked_mae_loss(y_pred[index1:index1+1], y_true[index1:index1+1]).item())
                    m_1.append(masked_mape_loss(y_pred[index1:index1+1], y_true[index1:index1+1]).item())
                    r_1.append(masked_mse_loss(y_pred[index1:index1+1], y_true[index1:index1+1]).item())
                    l_2.append(masked_mae_loss(y_pred[index2:index2+1], y_true[index2:index2+1]).item())
                    m_2.append(masked_mape_loss(y_pred[index2:index2+1], y_true[index2:index2+1]).item())
                    r_2.append(masked_mse_loss(y_pred[index2:index2+1], y_true[index2:index2+1]).item())
                    l_3.append(masked_mae_loss(y_pred[index3:index3+1], y_true[index3:index3+1]).item())
                    m_3.append(masked_mape_loss(y_pred[index3:index3+1], y_true[index3:index3+1]).item())
                    r_3.append(masked_mse_loss(y_pred[index3:index3+1], y_true[index3:index3+1]).item())
                    l_4.append(masked_mae_loss(y_pred[index4:index4+1], y_true[index4:index4+1]).item())
                    m_4.append(masked_mape_loss(y_pred[index4:index4+1], y_true[index4:index4+1]).item())
                    r_4.append(masked_mse_loss(y_pred[index4:index4+1], y_true[index4:index4+1]).item())
                    

                else:
                    loss_1 = self._compute_loss(y, output)
                    pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
                    true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)
                    loss = loss_1 + loss_g
                    # option
                    # loss = loss_1 + 10*loss_g
                    losses.append((loss_1.item()+loss_g.item()))

                    y_true = self.standard_scaler.inverse_transform(y)
                    y_pred = self.standard_scaler.inverse_transform(output)
                    mapes.append(masked_mape_loss(y_pred, y_true).item())
                    #rmses.append(masked_rmse_loss(y_pred, y_true).item())
                    mses.append(masked_mse_loss(y_pred, y_true).item())
                    
                    # Followed the DCRNN TensorFlow Implementation
                    l_1.append(masked_mae_loss(y_pred[index1:index1+1], y_true[index1:index1+1]).item())
                    m_1.append(masked_mape_loss(y_pred[index1:index1+1], y_true[index1:index1+1]).item())
                    r_1.append(masked_mse_loss(y_pred[index1:index1+1], y_true[index1:index1+1]).item())
                    l_2.append(masked_mae_loss(y_pred[index2:index2+1], y_true[index2:index2+1]).item())
                    m_2.append(masked_mape_loss(y_pred[index2:index2+1], y_true[index2:index2+1]).item())
                    r_2.append(masked_mse_loss(y_pred[index2:index2+1], y_true[index2:index2+1]).item())
                    l_3.append(masked_mae_loss(y_pred[index3:index3+1], y_true[index3:index3+1]).item())
                    m_3.append(masked_mape_loss(y_pred[index3:index3+1], y_true[index3:index3+1]).item())
                    r_3.append(masked_mse_loss(y_pred[index3:index3+1], y_true[index3:index3+1]).item())
                    l_4.append(masked_mae_loss(y_pred[index4:index4+1], y_true[index4:index4+1]).item())
                    m_4.append(masked_mape_loss(y_pred[index4:index4+1], y_true[index4:index4+1]).item())
                    r_4.append(masked_mse_loss(y_pred[index4:index4+1], y_true[index4:index4+1]).item())

                #if batch_idx % 100 == 1:
                #    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)
            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option
            
            duration1 = str(self.duration_list[0])
            duration2 = str(self.duration_list[1])
            duration3 = str(self.duration_list[2])
            duration4 = str(self.duration_list[3])
                
            # Followed the DCRNN PyTorch Implementation
            message = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
            self._logger.info(message)
                
            # Followed the DCRNN TensorFlow Implementation
            message = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration1, np.mean(l_1), np.mean(m_1),
                                                                                        np.sqrt(np.mean(r_1)))
            self._logger.info(message)

            message = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration2, np.mean(l_2), np.mean(m_2),
                                                                                        np.sqrt(np.mean(r_2)))
            self._logger.info(message)
            message = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration3, np.mean(l_3), np.mean(m_3),
                                                                                        np.sqrt(np.mean(r_3)))
            self._logger.info(message)
            message = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration4, np.mean(l_4), np.mean(m_4),
                                                                                        np.sqrt(np.mean(r_4)))
            self._logger.info(message)

            self._writer.add_scalar('demo loss', mean_loss, batches_seen)
            if label == 'without_regularization':
                return mean_loss, mean_mape, mean_rmse
            else:
                return mean_loss


    def _train(self, base_lr,
               steps, patience=200, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=0,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.GTS_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        self.evaluate(label='with_regularization', dataset='val', batches_seen=batches_seen, gumbel_soft=True)


    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
