'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-05-27 12:51:18
@LastEditTime: 2020-07-29 16:52:01
@Description:
'''
import numpy as np
import os.path as osp
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from mbrl.models.base import MLPRegression, CUDA, CPU, combined_shape, DataBuffer

DEFAULT_CONFIG = dict(
                n_ensembles=7,
                data_split=0.8,
                n_epochs=100,
                learning_rate=0.001,
                batch_size=256,
                hidden_sizes=(1024, 1024, 1024),
                buffer_size=500000,

                save=False,
                save_folder=None,
                load=False,
                load_folder=None,
                test_freq=2,
                test_ratio=0.1,
                activation="relu",
            )

class RegressionModelEnsemble:
    def __init__(self, input_dim, output_dim, config=DEFAULT_CONFIG):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_ensembles = config["n_ensembles"]
        # how many percentage of data in the training used for training one model, 0.8 = use 80% data to train
        self.data_split = config["data_split"] 
        self.n_epochs = config["n_epochs"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.test_freq = config["test_freq"]
        self.test_ratio = config["test_ratio"]
        activ = config["activation"]
        activ_f = nn.Tanh if activ == "tanh" else nn.ReLU
        self.data_buf = DataBuffer(self.input_dim, self.output_dim, max_len=config["buffer_size"])

        self.mu = torch.tensor(0.0)
        self.sigma = torch.tensor(1.0)
        self.label_mu = torch.tensor(0.0)
        self.label_sigma = torch.tensor(1.0)
        self.eps = 1e-3

        self.save = config["save"]
        if self.save:
            self.folder = config["save_folder"]
            if osp.exists(self.folder):
                print("Warning: Saving dir %s already exists! Storing model and buffer there anyway."%self.folder)
            else:
                os.makedirs(self.folder)
            self.data_buf_path = osp.join(self.folder, "dynamic_data_buf.pkl")
            self.model_path = osp.join(self.folder, "dynamic_model.pkl")
        if config["load"]:
            self.load_data(config["load_folder"])
        else:
            self.models = []
            for i in range(self.n_ensembles):
                self.models.append( CUDA(MLPRegression(self.input_dim, self.output_dim, 
                    config["hidden_sizes"], activation=activ_f)) )

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizers = []
        for i in range(self.n_ensembles):
            self.optimizers.append(torch.optim.Adam(self.models[i].parameters(), lr=self.lr))  
    
    def add_data_point(self, input_data, output_data):
        '''
        This method is used for streaming data setting, where one data will be added at each time.
        @param input_data [list or ndarray, (input_dim)]
        @param output_data [list or ndarray, (output_dim)]
        '''
        x = np.array(input_data).reshape(self.input_dim)
        y = np.array(output_data).reshape(self.output_dim)
        self.data_buf.store(x, y)

    def predict(self, data, var=False):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param data [list or ndarray or tensor, (batch, input_dim)]
        @return out [list or ndarray, (batch, output_dim)]
        '''
        inputs = data if torch.is_tensor(data) else torch.tensor(data).float()
        inputs = (inputs-self.mu) / (self.sigma)
        inputs = CUDA(inputs)
        with torch.no_grad():
            out_list = []
            for model in self.models:
                model.eval()
                out = model(inputs)
                out = CPU(out)
                out = out * (self.label_sigma) + self.label_mu
                out = out.numpy()
                out_list.append(out)
        out_mean = np.mean(out_list, axis=0)
        if var:
            out_var = np.var(out_list, axis=0)
            return out_mean, out_var
        return out_mean

    def predict_with_each_model(self, data):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param data [list or ndarray or tensor, (batch, n_ensembles, input_dim)]
        @return out [list or ndarray, (batch, n_ensembles, output_dim)]
        '''
        B, n, dim = data.shape
        assert n==self.n_ensembles, "the input data dimension is not correct!"
        inputs = data if torch.is_tensor(data) else torch.tensor(data).float()
        inputs = (inputs-self.mu) / (self.sigma)
        inputs = CUDA(inputs)
        with torch.no_grad():
            output = np.zeros((B, n, self.output_dim))
            for i in range(n):
                model = self.models[i]
                model.eval()
                out = model(inputs[:,i,:])
                out = CPU(out)
                out = out * (self.label_sigma) + self.label_mu
                out = out.numpy()
                output[:,i,:] = out
        return output

    def reset_dataset(self, new_dataset = None):
        # dataset format: list of [task_idx, state, action, next_state-state]
        if new_dataset is not None:
            self.dataset = new_dataset
        else:
            self.dataset = []
            
    def make_dataset(self, x, y, normalize = True):
        '''
        This method is used to generate dataset object for training.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''
        tensor_x = x if torch.is_tensor(x) else torch.tensor(x).float()
        tensor_y = y if torch.is_tensor(y) else torch.tensor(y).float()
        if normalize:
            self.mu = torch.mean(tensor_x, dim=0, keepdims=True)
            self.sigma = torch.std(tensor_x, dim=0, keepdims=True)
            self.label_mu = torch.mean(tensor_y, dim=0, keepdims=True)
            self.label_sigma = torch.std(tensor_y, dim=0, keepdims=True)      
            self.sigma[self.sigma<self.eps] = 1
            self.label_sigma[self.label_sigma<self.eps] = 1
            tensor_x = (tensor_x-self.mu) / (self.sigma)
            tensor_y = (tensor_y-self.label_mu) / (self.label_sigma)
        self.dataset = TensorDataset(tensor_x, tensor_y)
        return self.dataset

    def train_one_epoch(self, model, optimizer, dataloader):
        model.train()
        loss_train = 0
        for datas, labels in dataloader:
            datas = CUDA(datas)
            labels = CUDA(labels)
            optimizer.zero_grad()
            outputs = model(datas)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()*datas.shape[0] # sum of the loss
        loss_train /= len(dataloader.dataset)
        return loss_train

    def fit(self, x=None, y=None, use_data_buf=True, normalize=True):
        '''
        Train the model either from external data or internal data buf.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''
        # early stopping
        patience = 5
        best_loss = 1e3
        loss_increase = 0

        if use_data_buf:
            x, y = self.data_buf.get_all()
            dataset =  self.make_dataset(x, y, normalize = normalize)
        else: # use external data loader
            dataset = self.make_dataset(x, y, normalize = normalize)
        num_data = len(dataset)
        testing_len = int(self.test_ratio * num_data)
        training_len = num_data - testing_len

        train_set, test_set = random_split(dataset, [training_len, testing_len])
        test_loader = DataLoader(test_set, shuffle=True, batch_size=self.batch_size) 

        for epoch in range(self.n_epochs):
            # train each model
            train_losses = []
            for i in range(self.n_ensembles):
                train_data_num = len(train_set)
                split_len = int(self.data_split * train_data_num)
                train_set_single, _ = random_split(train_set, [split_len, train_data_num - split_len])
                train_loader = DataLoader(train_set_single, shuffle=True, batch_size=self.batch_size)
                loss = self.train_one_epoch(self.models[i], self.optimizers[i], train_loader)
                train_losses.append(loss)
                
            if (epoch+1) % self.test_freq == 0:
                train_loss_mean = np.mean(train_losses)
                train_loss_var = np.var(train_losses)
                if len(test_loader) > 0:
                    test_losses = []
                    for model in self.models:
                        test_losses.append(self.test_model(model, test_loader))
                    test_loss_mean = np.mean(test_losses)
                    test_loss_var = np.var(test_losses)
                    print(f"[{epoch}/{self.n_epochs}],loss train m: {train_loss_mean:.4f}, v: {train_loss_var:.4f}, test m: {test_loss_mean:.4f}, v: {test_loss_var:.4f}")
                else:
                    print(f"[{epoch}/{self.n_epochs}],mse train mean: {train_loss_mean:.4f}, var: {train_loss_var:.4f}, no testing data")
                if test_loss_mean < best_loss:
                    best_loss = test_loss_mean
                    loss_increase = 0
                else:
                    loss_increase += 1

                if loss_increase > patience:
                    break
        if self.save:
            self.save_data()

    def evaluate_rollout(self, states, actions, labels, eval_dim = None, debug=False):
        '''
        Calculate the mse loss along time steps
        ----------
            @param states [tensor, (T, batch, state dim)] : padded history states sequence
            @param actions [tensor, (T, batch, action dim)] : padded history actions sequence
            @param labels [tensor, (T, batch, state dim)] : padded history states_next
            @param length [tensor, (batch)] : length of each unpadded trajectory in the batch (must be sorted)
            @param eval_dim [slice or None] : determine which state dims will be evaluated. If None, evaluate all dims.
        ----------
            @return loss [list, (T - his_len + 1)]
        '''
        states = CUDA(states)
        actions = CUDA(actions)
        labels = CUDA(labels)
        
        T, batch_size, state_dim = states.shape

        rollouts = CUDA(torch.zeros((T, self.n_ensembles, batch_size, state_dim))) # [T, M, B, s]

        with torch.no_grad():
            for t in range(T):
                inputs = torch.cat((states[t], actions[t]), dim = 1)
                inputs = (inputs- CUDA(self.mu)) / CUDA(self.sigma)

                for i in range(self.n_ensembles):
                    outputs = self.models[i](inputs) # [B, s]
                    outputs = outputs * CUDA(self.label_sigma) + CUDA(self.label_mu)

                    rollouts[t, i] = outputs

        targets = labels # [T, B, s]

        if eval_dim is not None:
            targets = targets[:,:, eval_dim] #[T, B, dim]
            rollouts = rollouts[:,:,:,eval_dim] #[T, M, B, dim]

        MSE = torch.zeros( T, self.n_ensembles, batch_size) # [T, M, B]

        for i in range(self.n_ensembles):
            MSE[:,i,:] = torch.mean( (targets-rollouts[:,i])**2, dim=2 ) #[T, B]

        MSE = MSE.view(T, -1) # [T, M*B]
        loss_mean = torch.mean(MSE, dim=1) # [T]
        loss_mean = list(CPU(loss_mean).numpy())

        loss_var = torch.var(MSE, dim=1) # [T]
        loss_var = list(CPU(loss_var).numpy())

        return loss_mean, loss_var

    def test_model(self, model, testloader):
        '''
        Test the model with normalized test dataset.
        @param test_loader [torch.utils.data.DataLoader]
        '''
        model.eval()
        loss_test = 0
        for datas, labels in testloader:
            datas = CUDA(datas)
            labels = CUDA(labels)
            outputs = model(datas)
            loss = self.criterion(outputs, labels)
            loss_test += loss.item()*datas.shape[0]
        loss_test /= len(testloader.dataset)
        return loss_test

    def test(self, data, label):
        '''
        Test the model with unnormalized ndarray test dataset.
        @param data [list or ndarray, (batch, input_dim)]
        @param label [list or ndarray, (batch, output_dim)]
        '''
        pred_mean = self.predict(data, var=False) 
        #mse = np.mean((pred-label)**2)
        #print("MSE: ", mse)
        pred_mean = CUDA(torch.tensor(pred_mean).float())
        labels = CUDA(label) if torch.is_tensor(label) else CUDA(torch.tensor(label).float())
        loss = self.criterion(pred_mean, labels)
        return loss.item()

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)

        for model in self.models:
            model.apply(weight_reset)

    def save_model(self, path):
        
        checkpoint = {"n_ensembles":self.n_ensembles, "mu":self.mu, "sigma":self.sigma, "label_mu":self.label_mu, "label_sigma":self.label_sigma}
        for i in range(self.n_ensembles):
            name = "model_state_dict_"+str(i)
            checkpoint[name] = self.models[i]
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.n_ensembles = checkpoint["n_ensembles"]
        self.models = []
        for i in range(self.n_ensembles):
            name = "model_state_dict_"+str(i)
            model = checkpoint[name]
            self.models.append(CUDA(model))
        self.mu = checkpoint["mu"]
        self.sigma = checkpoint["sigma"]
        self.label_mu = checkpoint["label_mu"]
        self.label_sigma = checkpoint["label_sigma"]

    def save_data(self):
        self.save_model(self.model_path)
        self.data_buf.save(self.data_buf_path)
        print("Successfully save model and data buffer to %s"%self.folder)

    def load_data(self, path):
        model_path = osp.join(path, "dynamic_model.pkl")
        if osp.exists(model_path):
            self.load_model(model_path)
            print("Loading dynamic model from %s ."%model_path)
        else:
            print("We can not find the model from %s"%model_path)
        data_buf_path = osp.join(path, "dynamic_data_buf.pkl")
        if osp.exists(data_buf_path):
            print("Loading dynamic data buffer from %s ."%data_buf_path)
            self.data_buf.load(data_buf_path)
        else:
            print("We can not find the dynamic data buffer from %s"%data_buf_path)
