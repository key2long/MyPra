import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data_utils import PRAData
import torch.optim as optim

learning_rate = 0.001
batch_size = 8
train_path = './train_data.txt'
test_path = ''
validation_path = ''
input_size = 1620
num_classes = 2
epoch_num = 200

train_data = PRAData(train_path)
train_loader = DataLoader(train_data, batch_size=batch_size)

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        output = self.linear(x)
        return output

model = LogisticRegression(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    train_loss_sum = []
    for epoch in range(epoch_num):
        print('epoch is %d:' % epoch)
        for path_feature, label in train_loader:
            #print(path_feature.dtype, label.dtype)
            optimizer.zero_grad()
            outputs = model(path_feature)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss_sum.append(loss.item())
