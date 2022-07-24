import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss

###test code for calculating mape, rmse
from lib.metrics import masked_mae_np, masked_mape_np, masked_rmse_np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNN_Demo_Supervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._model_kwargs.get('max_grad_norm', 1.)

        #get model_dir
        self.model_dir = self._data_kwargs.get('model_dir')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_demo_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 2))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder
        self.index_list = self._train_kwargs.get('index_list')
        self.duration_list = self._train_kwargs.get('duration_list')

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
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
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
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
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            demo_iterator = self._data['demo_loader'].get_iterator()

            for _, (x, y) in enumerate(demo_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            demo_iterator = self._data['demo_loader'].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            #list for storing forecasts
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

            #evaluate for demo dataset
            demo_losses = []
            demo_y_truths = []
            demo_y_preds = []

            for _, (x,y) in enumerate(demo_iterator):
                x, y = self._prepare_data(x, y)

                demo_output = self.dcrnn_model(x)
                demo_loss = self._compute_loss(y, demo_output)
                demo_losses.append(demo_loss.item())

                demo_y_truths.append(y.cpu())
                demo_y_preds.append(demo_output.cpu())   

                #add data by horizon 
                demo_y_truth_scaled = self.standard_scaler.inverse_transform(y)
                demo_y_pred_scaled = self.standard_scaler.inverse_transform(demo_output)

                #indexes 
                index1 = self.index_list[0]
                index2 = self.index_list[1]
                index3 = self.index_list[2]
                index4 = self.index_list[3]

                #horizon 1 
                l_1.append(masked_mae_np(preds=demo_y_pred_scaled[index1:index1+1].cpu().numpy(), labels=demo_y_truth_scaled[index1:index1+1].cpu().numpy(), null_val=0))
                m_1.append(masked_mape_np(preds=demo_y_pred_scaled[index1:index1+1].cpu().numpy(), labels=demo_y_truth_scaled[index1:index1+1].cpu().numpy(), null_val=0))
                r_1.append(masked_rmse_np(preds=demo_y_pred_scaled[index1:index1+1].cpu().numpy(), labels=demo_y_truth_scaled[index1:index1+1].cpu().numpy(), null_val=0))
                l_2.append(masked_mae_np(preds=demo_y_pred_scaled[index2:index2+1].cpu().numpy(), labels=demo_y_truth_scaled[index2:index2+1].cpu().numpy(), null_val=0))
                m_2.append(masked_mape_np(preds=demo_y_pred_scaled[index2:index2+1].cpu().numpy(), labels=demo_y_truth_scaled[index2:index2+1].cpu().numpy(), null_val=0))
                r_2.append(masked_rmse_np(preds=demo_y_pred_scaled[index2:index2+1].cpu().numpy(), labels=demo_y_truth_scaled[index2:index2+1].cpu().numpy(), null_val=0))
                l_3.append(masked_mae_np(preds=demo_y_pred_scaled[index3:index3+1].cpu().numpy(), labels=demo_y_truth_scaled[index3:index3+1].cpu().numpy(), null_val=0))
                m_3.append(masked_mape_np(preds=demo_y_pred_scaled[index3:index3+1].cpu().numpy(), labels=demo_y_truth_scaled[index3:index3+1].cpu().numpy(), null_val=0))
                r_3.append(masked_rmse_np(preds=demo_y_pred_scaled[index3:index3+1].cpu().numpy(), labels=demo_y_truth_scaled[index3:index3+1].cpu().numpy(), null_val=0)) 
                l_4.append(masked_mae_np(preds=demo_y_pred_scaled[index4:index4+1].cpu().numpy(), labels=demo_y_truth_scaled[index4:index4+1].cpu().numpy(), null_val=0))
                m_4.append(masked_mape_np(preds=demo_y_pred_scaled[index4:index4+1].cpu().numpy(), labels=demo_y_truth_scaled[index4:index4+1].cpu().numpy(), null_val=0))
                r_4.append(masked_rmse_np(preds=demo_y_pred_scaled[index4:index4+1].cpu().numpy(), labels=demo_y_truth_scaled[index4:index4+1].cpu().numpy(), null_val=0))                

            demo_mean_loss = np.mean(demo_losses)
            self._writer.add_scalar('Demo loss', demo_loss, batches_seen)

            demo_y_preds = np.concatenate(demo_y_preds, axis=1)
            demo_y_truths = np.concatenate(demo_y_truths, axis=1)
            
            demo_y_truths_scaled = []
            demo_y_preds_scaled = []
            for t in range(demo_y_preds.shape[0]):
                demo_y_truth = self.standard_scaler.inverse_transform(demo_y_truths[t])
                demo_y_pred =  self.standard_scaler.inverse_transform(demo_y_preds[t])
                demo_y_truths_scaled.append(demo_y_truth)
                demo_y_preds_scaled.append(demo_y_pred)          

            ##possible code to calculate MAE, MAPE, RMSE for test dataset by horizon
            mae = masked_mae_np(preds=demo_y_preds_scaled, labels=demo_y_truths_scaled, null_val=0)
            rmse = masked_rmse_np(preds=demo_y_preds_scaled, labels=demo_y_truths_scaled, null_val=0)
            mape = masked_mape_np(preds=demo_y_preds_scaled, labels=demo_y_truths_scaled, null_val=0)
            print(f"Test: mae {mae} rmse {rmse} mape {mape}")

            #duration info 
            duration1 = str(self.duration_list[0])
            duration2 = str(self.duration_list[1])
            duration3 = str(self.duration_list[2])
            duration4 = str(self.duration_list[3])

            #calculate data by horizon 
            message_h1 = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration1, np.mean(l_1), np.mean(m_1),
                                                                                           np.mean(r_1))
            self._logger.info(message_h1)
            message_h2 = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration2, np.mean(l_2), np.mean(m_2),
                                                                                           np.mean(r_2))
            self._logger.info(message_h2)
            message_h3 = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration3, np.mean(l_3), np.mean(m_3),
                                                                                           np.mean(r_3))
            self._logger.info(message_h3)
            message_h6 = 'Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(duration4, np.mean(l_4), np.mean(m_4),
                                                                                           np.mean(r_4))
            self._logger.info(message_h6)

            return 

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)


        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        self.evaluate(dataset='demo', batches_seen=batches_seen)

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
