'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-05-27 12:51:18
@LastEditTime: 2020-07-29 15:22:42
@Description:
'''
import numpy as np
import os.path as osp
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from mbrl.models.base import MLPRegression, MLPCategorical, CUDA, CPU, combined_shape, DataBuffer

DEFAULT_CONFIG = dict(
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

class RegressionModel:
    def __init__(self, input_dim, output_dim, config=DEFAULT_CONFIG):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

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
            #self.save_freq = config["save_freq"]
            #self.save_path = config["save_path"]
        if config["load"]:
            self.load_data(config["load_folder"])
        else:
            self.model = CUDA(MLPRegression(self.input_dim, self.output_dim, config["hidden_sizes"], activation=activ_f))

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        
    
    def add_data_point(self, input_data, output_data):
        '''
        This method is used for streaming data setting, where one data will be added at each time.
        @param input_data [list or ndarray, (input_dim)]
        @param output_data [list or ndarray, (output_dim)]
        '''
        x = np.array(input_data).reshape(self.input_dim)
        y = np.array(output_data).reshape(self.output_dim)
        self.data_buf.store(x, y)
        
    def predict(self, data):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param data [list or ndarray or tensor, (batch, input_dim)]
        @return out [list or ndarray, (batch, output_dim)]
        '''
        self.model.eval()
        inputs = data if torch.is_tensor(data) else torch.tensor(data).float()
        inputs = (inputs-self.mu) / (self.sigma)
        inputs = CUDA(inputs)
        with torch.no_grad():
            out = self.model(inputs)
            out = CPU(out)
            out = out * (self.label_sigma) + self.label_mu
            out = out.numpy()
        return out

    def reset_dataset(self, new_dataset = None):
        # dataset format: list of [task_idx, state, action, next_state-state]
        if new_dataset is not None:
            self.dataset = new_dataset
        else:
            self.dataset = []
            
    def make_dataloader(self, x, y, normalize = True):
        '''
        This method is used to generate dataloader object for training.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''

        tensor_x = x if torch.is_tensor(x) else torch.tensor(x).float()

        tensor_y = y if torch.is_tensor(y) else torch.tensor(y).float()
        num_data = tensor_x.shape[0]

        if normalize:
            self.mu = torch.mean(tensor_x, dim=0, keepdims=True)
            self.sigma = torch.std(tensor_x, dim=0, keepdims=True)
            self.label_mu = torch.mean(tensor_y, dim=0, keepdims=True)
            self.label_sigma = torch.std(tensor_y, dim=0, keepdims=True)
            
            self.sigma[self.sigma<self.eps] = 1
            self.label_sigma[self.label_sigma<self.eps] = 1

            print("data normalized")
            print("mu: ", self.mu)
            print("sigma: ", self.sigma)
            print("label mu: ", self.label_mu)
            print("label sigma: ", self.label_sigma)

            tensor_x = (tensor_x-self.mu) / (self.sigma)
            tensor_y = (tensor_y-self.label_mu) / (self.label_sigma)

        dataset = TensorDataset(tensor_x, tensor_y)

        testing_len = int(self.test_ratio * num_data)
        training_len = num_data - testing_len

        train_set, test_set = random_split(dataset, [training_len, testing_len])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=self.batch_size)

        return train_loader, test_loader

    def fit(self, x=None, y=None, use_data_buf=True, normalize=True):
        '''
        Train the model either from external data or internal data buf.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''

        # early stopping
        patience = 6
        best_loss = 1e3
        loss_increase = 0

        if use_data_buf:
            x, y = self.data_buf.get_all()
            train_loader, test_loader =  self.make_dataloader(x, y, normalize = normalize)
        else: # use external data loader
            train_loader, test_loader = self.make_dataloader(x, y, normalize = normalize)
        
        for epoch in range(self.n_epochs):
            self.model.train()
            loss_train = 0
            loss_test = 1e5
            for datas, labels in train_loader:
                datas = CUDA(datas)
                labels = CUDA(labels)
                self.optimizer.zero_grad()

                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()*datas.shape[0] # sum of the loss
            
            # if self.save and (epoch+1) % self.save_freq == 0:
            #     self.save_model(self.save_path)
                
            if (epoch+1) % self.test_freq == 0:
                loss_train /= len(train_loader.dataset)
                loss_test = -0.1234
                if len(test_loader) > 0:
                    loss_test = self.test_model(test_loader)
                    print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}, loss test  {loss_test:.4f}")
                else:
                    print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}, no testing data")
                loss_unormalized = self.test(x[::50], y[::50])
                print("loss unnormalized: ", loss_unormalized)

                if loss_test < best_loss:
                    best_loss = loss_test
                    loss_increase = 0
                else:
                    loss_increase += 1

                if loss_increase > patience:
                    break
                #print(loss_increase)

            # #if self.save and (epoch+1) % self.save_freq == 0 and loss_test == best_loss:
            # if self.save and loss_test <= best_loss:
            #     self.save_model(self.save_path)
            #     print("Saving model..... with loss", loss_test)
        
        if self.save:
            self.save_data()

    def evaluate_rollout(self, states, actions, labels, eval_dim = None, debug=False):
        '''
        Calculate the mse loss along time steps
        ----------
            @param states [tensor, (T, batch, state dim)] : padded history states sequence
            @param actions [tensor, (T, batch, action dim)] : padded history actions sequence
            @param labels [tensor, (T, batch, label dim)] : padded history states_next
            @param history_len [int] : use history-len data as the input to rollout other steps
            @param length [tensor, (batch)] : length of each unpadded trajectory in the batch (must be sorted)
            @param eval_dim [slice or None] : determine which state dims will be evaluated. If None, evaluate all dims.
        ----------
            @return loss [list, (T - his_len + 1)]
        '''
        states = CUDA(states)
        actions = CUDA(actions)
        labels = CUDA(labels)
        
        T, batch_size, state_dim = states.shape

        rollouts = CUDA(torch.zeros(labels.shape)) # [T, B, s]

        with torch.no_grad():
            for t in range(T):
                inputs = torch.cat((states[t], actions[t]), dim = 1)
                inputs = (inputs- CUDA(self.mu)) / CUDA(self.sigma)
                outputs = self.model(inputs) # [B, s]
                outputs = outputs * CUDA(self.label_sigma) + CUDA(self.label_mu)
                rollouts[t] = outputs

        targets = labels # [T, B, s]

        if eval_dim is not None:
            targets = targets[:,:, eval_dim] #[steps, B, dim]
            rollouts = rollouts[:,:,eval_dim] #[steps, B, dim]

        # print(eval_dim)
        # print(torch.mean(abs(targets[:,:] - rollouts[:,:])))
        #print(targets[2,:] - rollouts[2,:])

        MSE = torch.mean( (targets-rollouts)**2, dim=2 ) #[steps, B]

        loss_mean = torch.mean(MSE, dim=1) # [steps]
        loss_mean = list(CPU(loss_mean).numpy())

        loss_var = torch.var(MSE, dim=1) # [steps]
        loss_var = list(CPU(loss_var).numpy())

        return loss_mean, loss_var

    def enable_dropout(self):
        self.model.train()

    def disable_dropout(self):
        self.model.eval()

    def test_model(self, testloader):
        '''
        Test the model with normalized test dataset.
        @param test_loader [torch.utils.data.DataLoader]
        '''
        self.model.eval()
        loss_test = 0
        for datas, labels in testloader:
            datas = CUDA(datas)
            labels = CUDA(labels)
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_test += loss.item()*datas.shape[0]
        loss_test /= len(testloader.dataset)
        self.model.train()
        return loss_test

    def test(self, data, label):
        '''
        Test the model with unnormalized ndarray test dataset.
        @param data [list or ndarray, (batch, input_dim)]
        @param label [list or ndarray, (batch, output_dim)]
        '''
        pred = self.predict(data) 
        #mse = np.mean((pred-label)**2)
        #print("MSE: ", mse)
        pred = CUDA(torch.tensor(pred).float())
        labels = CUDA(torch.tensor(label).float())
        loss = self.criterion(pred, labels)
        return loss.item()

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        
        checkpoint = {"model_state_dict":self.model, "mu":self.mu, "sigma":self.sigma, "label_mu":self.label_mu, "label_sigma":self.label_sigma}
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model_state_dict"]
        self.model = CUDA(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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


class Classifier:
    def __init__(self, input_dim, output_dim, config):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_epochs = config["n_epochs"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.save = config["save"]
        self.save_freq = config["save_freq"]
        self.save_path = config["save_path"]
        self.test_freq = config["test_freq"]
        self.test_ratio = config["test_ratio"]
        activ = config["activation"]
        activ_f = nn.Tanh if activ.lower() == "tanh" else nn.ReLU

        self.mu = torch.tensor(0.0)
        self.sigma = torch.tensor(1.0)
        self.eps = 0.001

        if config["load"]:
            self.load_model(config["load_path"])
        else:
            self.model = CUDA(MLPCategorical(self.input_dim, self.output_dim, config["hidden_sizes"], activation=activ_f))

        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.data_buf = DataBuffer(self.input_dim, self.output_dim, max_len=config["buffer_size"])
    
    def add_data_point(self, input_data, output_data):
        '''
        This method is used for streaming data setting, where one data will be added at each time.
        @param input_data [list or ndarray, (input_dim)]
        @param output_data [list or ndarray, (output_dim)]
        '''
        x = np.array(input_data).reshape(self.input_dim)
        y = np.array(output_data).reshape(self.output_dim)
        self.data_buf.store(x, y)
        
    def predict(self, data):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param data [list or ndarray, (batch, input_dim)]
        @return out [list or ndarray, (batch, output_dim)]
        '''
        self.model.eval()
        inputs = torch.tensor(data).float()
        inputs = (inputs-self.mu) / (self.sigma)
        inputs = CUDA(inputs)
        with torch.no_grad():
            out = self.model(inputs)
            out = CPU(out.argmax(dim=1, keepdim=True))  # get the index, [batch, 1])
            out = out.numpy()
        return out
            
    def make_dataloader(self, x, y, normalize = True):
        '''
        This method is used to generate dataloader object for training.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''
        
        tensor_x = torch.tensor(x).float()
        tensor_y = torch.tensor(y).long()
        num_data = tensor_x.shape[0]

        if normalize:
            self.mu = torch.mean(tensor_x, dim=0, keepdims=True)
            self.sigma = torch.std(tensor_x, dim=0, keepdims=True)
            self.sigma[self.sigma<self.eps] = 1
            print("data normalized")
            print("mu: ", self.mu)
            print("sigma: ", self.sigma)
            tensor_x = (tensor_x-self.mu) / (self.sigma)

        dataset = TensorDataset(tensor_x, tensor_y)

        testing_len = int(self.test_ratio * num_data)
        training_len = num_data - testing_len
        train_set, test_set = random_split(dataset, [training_len, testing_len])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        return train_loader, test_loader

    def fit(self, x=None, y=None, use_data_buf=True, normalize=True):
        '''
        Train the model either from external data or internal data buf.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''
        if use_data_buf:
            x, y = self.data_buf.get_all()
            train_loader, test_loader =  self.make_dataloader(x, y, normalize = normalize)
        else: # use external data loader
            train_loader, test_loader = self.make_dataloader(x, y, normalize = normalize)
        
        for epoch in range(self.n_epochs):
            loss_train, acc_train = 0, 0
            self.model.train()
            for datas, labels in train_loader:
                datas = CUDA(datas)
                labels = CUDA(labels)

                self.optimizer.zero_grad()
                outputs = self.model(datas)
                labels = torch.squeeze(labels)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()*datas.shape[0] # sum of the loss
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index
                acc_train += pred.eq(labels.view_as(pred)).sum().item()

            if self.save and (epoch+1) % self.save_freq == 0:
                self.save_model(self.save_path)
                
            if (epoch+1) % self.test_freq == 0:
                loss_train /= len(train_loader.dataset)
                acc_train /= len(train_loader.dataset)
                loss_test, acc_test = -0.1234, -0.12
                if len(test_loader) > 0:
                    loss_test, acc_test = self.test_model(test_loader)
                    print(f"epoch[{epoch}/{self.n_epochs}],train l|acc: {loss_train:.4f}|{100.*acc_train:.2f}%, test l|acc  {loss_test:.4f}|{100.*acc_test:.2f}%")
                else:
                    print(f"epoch[{epoch}/{self.n_epochs}],train l|acc: {loss_train:.4f}|{100.*acc_train:.2f}%")
        
        if self.save:
            self.save_model(self.save_path)

    def test_model(self, testloader):
        '''
        Test the model with normalized test dataset.
        @param test_loader [torch.utils.data.DataLoader]
        '''
        self.model.eval()
        loss_test, acc_test = 0, 0
        for datas, labels in testloader:
            datas = CUDA(datas)
            labels = CUDA(labels)
            outputs = self.model(datas)
            labels = torch.squeeze(labels)
            loss = self.criterion(outputs, labels)
            loss_test += loss.item()*datas.shape[0]
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc_test += pred.eq(labels.view_as(pred)).sum().item()
        loss_test /= len(testloader.dataset)
        acc_test /= len(testloader.dataset)
        self.model.train()
        return loss_test, acc_test

    def test(self, data, label):
        '''
        Test the model with unnormalized ndarray test dataset.
        @param data [list or ndarray, (batch, input_dim)]
        @param label [list or ndarray, (batch, output_dim)]
        '''
        pred = self.predict(data) 
        pred = CUDA(torch.tensor(pred).long())
        labels = CUDA(torch.tensor(label).long())
        acc = pred.eq(labels.view_as(pred)).mean().item()
        return acc

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        
        checkpoint = {"model_state_dict":self.model, "mu":self.mu, "sigma":self.sigma}
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model_state_dict"]
        self.model = CUDA(self.model)
        self.mu = checkpoint["mu"]
        self.sigma = checkpoint["sigma"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)