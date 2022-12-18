#PyTorch Implementation
import torch
import torch.nn as nn

#Base Model
class CNN_LSTM_Model(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  
        x = self.conv1d(x)  
        x = self.act1(x)
        x = self.maxPool(x)  
        x = self.drop(x)
        x = x.transpose(-1, -2)  
        x, (_, _) = self.lstm(x)  
        x = self.act2(x)
        x = x.squeeze(dim=1) 
        x = self.cls(x)
        x = self.act4(x)
        return x

#Efficient Chanel Attention
class CNN_LSTM_Model_ECA(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNN_LSTM_Model_ECA, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
        self.act3 = nn.Sigmoid()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  
        x = self.conv1d(x)  
        x = self.act1(x)
        x = self.maxPool(x)  
        x = self.drop(x)
        x = x.transpose(-1, -2)  
        x, (_, _) = self.lstm(x)  
        x = self.act2(x)
        x = x.squeeze(dim=1) 
        attn = self.attn(x)  
        attn = self.act3(attn)
        x = x * attn
        x = self.cls(x)
        x = self.act4(x)
        return x

#Channel Attention Module
class CNN_LSTM_Model_CAM(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNN_LSTM_Model_CAM, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.cam_fc = nn.Linear(window, window)

    def forward(self, x):
        x = x.transpose(-1, -2)  
        x = self.conv1d(x)  
        x = self.act1(x)

        avg = x.mean(dim=1)  
        cam_attn = self.se_fc(avg).softmax(dim=-1)  
        x = torch.einsum("bnd,bd->bnd", x, cam_attn)

        x = self.maxPool(x)  
        x = self.drop(x)
        x = x.transpose(-1, -2)  
        x, (_, _) = self.lstm(x)  
        x = self.act2(x)
        x = x.squeeze(dim=1) 
        x = self.cls(x)
        x = self.act4(x)
        return x
 
#Spatial Attention Module
class CNNLSTMModel_SAM(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNN_LSTM_Model_SAM, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.sam_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  
        x = self.conv1d(x)  
        x = self.act1(x)

        avg = x.mean(dim=2)  
        sam_attn = self.hw_fc(avg).softmax(dim=-1)  
        x = torch.einsum("bnd,bn->bnd", x, sam_attn)

        x = self.maxPool(x)  
        x = self.drop(x)
        x = x.transpose(-1, -2)  
        x, (_, _) = self.lstm(x) 
        x = self.act2(x)
        x = x.squeeze(dim=1) 
        x = self.cls(x)
        x = self.act4(x)
        return x



class CNN_LSTM_Model_CBAM(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNN_LSTM_Model_CBAM, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.cam_fc = nn.Linear(window, window)
        self.sam_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  
        x = self.conv1d(x)  
        x = self.act1(x)

        avg = x.mean(dim=1)  
        cam_attn = self.se_fc(avg).softmax(dim=-1)  
        x = torch.einsum("bnd,bd->bnd", x, cam_attn)
        
        avg = x.mean(dim=2)  
        sam_attn = self.hw_fc(avg).softmax(dim=-1)  
        x = torch.einsum("bnd,bn->bnd", x, sam_attn)

        x = self.maxPool(x) 
        x = self.drop(x)
        x = x.transpose(-1, -2)  
        x, (_, _) = self.lstm(x)  
        x = self.act2(x)
        x = x.squeeze(dim=1) 
        x = self.cls(x)
        x = self.act4(x)
        return x


