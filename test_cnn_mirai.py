import pandas
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from time import time
import numpy as np

#--------------------------------------------------------------------------------------------------------------
# read data from dataset
#nhan cac feature
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
kdd_data_10percent = pandas.read_csv("train_label.out", header=None, names = col_names)




col_names1 = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]
kdd_data_10percent_nolabel = pandas.read_csv("train_nolabel.out", header=None, names = col_names1)



#print('du lieu input:')
#print (kdd_data_10percent.iloc[1,:])
#print (len(kdd_data_10percent.iloc[:,1]))
#print kdd_data_10percent.describe()
#print kdd_data_10percent['label'].value_counts()
num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
features = kdd_data_10percent[num_features].astype(float)
#print features.describe()
label_cnn =[]
label_cnn = kdd_data_10percent['label'].as_matrix()


#------------------------------------------------------------------------------------------------------
# Training settings for pytorch
parser = argparse.ArgumentParser(description='PyTorch KDD99 Example')
parser.add_argument('--no_download_data', action='store_true', default=False,
                    help='Do not download data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


torch.backends.cudnn.benchmark=True

print("Number of epochs: ", args.epochs)
print("Batch size: ", args.batch_size)
print("Log interval: ", args.log_interval)
print("Learning rate: ", args.lr)
print(" Cuda: ", args.cuda)

#------------------------------------------------------------------------------------------------------
#model CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(41,50)
        # self.Linear2 = nn.Linear(10,50)
        torch.nn.init.kaiming_uniform(self.Linear1.weight)
        torch.nn.init.constant(self.Linear1.bias,0.1)
        # self.fc1 = nn.Linear(50, 10)
        # self.fc2 = nn.Linear(10, 10)
        # self.fc3 = nn.Linear(10, 40)

    def forward(self, x):
        print(x.size())
        x = self.Linear1(x)
        # x = F.relu(x)
        # x = self.Linear2(x)
        # x = F.relu(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = F.log_softmax(x)
        return x

#--------------------------------------------------------------------------------------------------
# training process

# run with cuda if have
model = Net()
args.cuda = True
if args.cuda:
    model.cuda()

#optimizer model cnn
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#convert dataframe to numpy data
input_model_data_numpy=kdd_data_10percent_nolabel.values
input_model_data = torch.from_numpy(input_model_data_numpy).float()

input_model_label_numpy = label_cnn
input_model_label = torch.from_numpy(input_model_label_numpy)




# get batch
def get_batch(data1,data2):
    #print('do dai data '+str(len(data)/64))
    for i in range(0,len(data1)/args.batch_size+1):
        tmp_data1 = data1[i*args.batch_size: (i+1)*args.batch_size]
        tmp_data2 = data2[i*args.batch_size: (i+1)*args.batch_size]
        yield (tmp_data1, tmp_data2)




# du lieu cho train
for epoch in range(args.epochs):
    batchs = get_batch(input_model_data, input_model_label)

    total =0
    losses=[]
    idbatch =0
    for batch_idx, out_for in enumerate(batchs):
        #print(type(batch))
        idbatch+=1
        batch = out_for[0].cuda()
        target = out_for[1].cuda()

        data, target = Variable(batch), Variable(target)
        #print(batch)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        #print(loss.data)
        losses.append(loss.data[0])    
        #print(sum(losses))
        #print("\nTrain set batch {}: Average loss: {:.4f}".format(idbatch,sum(losses) / len(losses)))
    # count acc training process
        correct = 0
        pred = output.data.max(1)[1].cpu().numpy()
        label = target.data.cpu().numpy()
        a=len(label)
        b=(np.sum(pred == label))
        #print(' acc training = {:.4f}'.format(float(b)/a))
        total +=float(b)/a
    print('Clgt!!')
    print('acc training for epoch: {} have avg acc = {:.8f}'.format(epoch,total/idbatch))

print('Training completed! ')
print(total/idbatch)




#-----------------------------------------------------
#test process
model.eval()
test_loss = 0
correct = 0
#load data test
kdd_data_test = pandas.read_csv("kddinput.out", header=None, names = col_names1)

kdd_data_test_nolabel = pandas.read_csv("kddinput.out", header=None, names = col_names1)

label_cnn =[]
label_cnn = kdd_data_test['duration'].as_matrix()

#convert dataframe to numpy data
input_model_data_numpy_test=kdd_data_test_nolabel.values
input_model_data = torch.from_numpy(input_model_data_numpy_test).float()

input_model_label_numpy = label_cnn
input_model_label_test = torch.from_numpy(input_model_label_numpy)
print (input_model_label_test.size())




total =0
losses=[]
idbatch =0
for batch_idx, out_for in enumerate(batchs):
    #print(type(batch))
    idbatch+=1
    batch = out_for[0].cuda()
    target = out_for[1].cuda()
    data, target = Variable(batch), Variable(target)
    data = Variable([data[0]])
    data = data[0].cuda()
    #
    # print(data)
    output = model(data)
    pred = output.data.max(1)[1].cpu().numpy()
    #print('output check:')
    #print(pred)
    #label = target.data.cpu().numpy()
    #a=len(label)
    #b=(np.sum(pred == label))
    #print(' acc training = {:.4f}'.format(float(b)/a))
    #print(float(b)/a)
    #total +=float(b)/a
#print('Testing completed! with acc=')
#print(total/idbatch)
#loss = F.nll_loss(output, target)
#print("loss testing: "+str(loss.data[0]))
#print('\nTest set: Average loss: {:.4f}\n'.format(loss.data[0]))



print('model: ')
print(model)
