from torch.utils.data import DataLoader, Dataset
import numpy as np


def load_all(data_path):
    feature = []
    label = []
    with open(data_path, 'r') as f:
        datas = f.readlines()
        for data in datas:
            data = eval(data.strip().split('\t')[1])
            for n, d in enumerate(data):
                data[n] = float(d)
            if len(data) == 1621:
                feature.append(np.array(data[1:]))
                label.append(np.array(data[0]))
    feature = np.array(feature, dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    return feature, label


class PRAData(Dataset):
    def __init__(self, data_path):
        super(PRAData, self).__init__()
        self.feature, self.label = load_all(data_path)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


if __name__ == '__main__':
    train_data = PRAData('./train_data.txt')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    print(type(train_loader))
    dataiter = iter(train_loader)
    data, label = dataiter.next()
    print(type(data), type(label))
    print(data.size(), label.size(), type(data[0][0]))