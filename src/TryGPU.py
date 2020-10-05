import os, sys
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import multiprocessing.pool
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def handle_error(error: str):
    print(error)
    sys.stdout.flush()


class MhYaoPRA(nn.Module):
    def __init__(self,
                 meta_path_feature_size: int,
                 num_relations: int):
        super(MhYaoPRA, self).__init__()
        classifiers = torch.randn((num_relations, meta_path_feature_size), requires_grad=True)
        self.classifiers = torch.nn.Parameter(classifiers)
        self.sigmod = nn.Sigmoid()
        self.register_parameter("classifiers", self.classifiers)

    def forward(self,
                batch_features_with_rid: torch.tensor):
        """

        :param batch_features_with_rid: (rid, batch_features_of_rid)
                                         where [batch_features_of_rid]_{1*batch_size*feature_size}
        :return: results
        """
        # print(f"input shape:{batch_features_with_rid.shape}")
        # print(f"classifier shape:{self.classifiers.shape}")
        rid = int(batch_features_with_rid[0, 0, 0])
        scores = torch.matmul(batch_features_with_rid[0, :, 1:], self.classifiers[rid, :])
        results = self.sigmod(scores)
        return results


class MhYaoPRAData(Dataset):
    def __init__(self, data, label):
        super(MhYaoPRAData, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


def generate_fake_data(args: argparse,
                       train_valid_test: str):
    e_list = ["E"+str(i) for i in range(args.entity_num)]
    r_list = ["R" + str(i) for i in range(args.relation_num)]
    facts_list = []
    facts_num = 0
    if train_valid_test == "train":
        facts_num = args.train_facts_num
    elif train_valid_test == "valid":
        facts_num = args.valid_facts_num
    elif train_valid_test == "test":
        facts_num = args.test_facts_num
    for fact_idx in range(facts_num):
        head_idx = np.random.choice(args.entity_num)
        relation_idx = np.random.choice(args.relation_num)
        tail_idx = np.random.choice(args.entity_num)
        tmp_fact = [e_list[head_idx], r_list[relation_idx], e_list[tail_idx]]
        facts_list.append(tmp_fact)
    return facts_list, e_list, r_list


def assign_labels():
    pass


def get_train_matrix(args: argparse,
                     r_idx: int,
                     model_pt: MhYaoPRA):
    labels = []
    train_matrix = torch.randn((2*args.pos_sample_size, args.feature_size + 1))
    train_matrix[:,0] = r_idx
    # lables_tensor = torch.from_numpy(np.array(labels)).float()
    lables_tensor = model_pt(train_matrix.unsqueeze(0)).detach()
    for i in range(2*args.pos_sample_size):
        if lables_tensor[i] > 0.5:
            lables_tensor[i] = 1.0
        else:
            lables_tensor[i] = 0.0
    return train_matrix, lables_tensor


def get_train_feature_matrix_with_rid(args: argparse,
                                      r_list: list,
                                      model_pt: MhYaoPRA):
    train_data_list = []
    train_label_list = []
    print(f"为每个relation生成fake Train数据。")
    for r in tqdm(r_list):
        tmp_train_matrix, tmp_labels_tensor = get_train_matrix(args, r_list.index(r), model_pt)
        train_data_list.append(tmp_train_matrix)
        train_label_list.append(tmp_labels_tensor)
    return train_data_list, train_label_list


def LoadTrainGraph(args: argparse,
                   model_pt: MhYaoPRA):
    facts_list, e_list, r_list = generate_fake_data(args, "train")
    pos_sample_dict = {}
    for fact in facts_list:
        tmp_r = fact[1]
        if tmp_r not in pos_sample_dict:
            pos_sample_dict[tmp_r] = [fact]
        else:
            pos_sample_dict[tmp_r].append(fact)
    # args.pos_sample_size = min([len(pos_sample_dict[r]) for r in pos_sample_dict.keys()])
    train_feature_matrix_with_rid_and_labels = get_train_feature_matrix_with_rid(args, r_list, model_pt)
    return train_feature_matrix_with_rid_and_labels


def get_feature_vec(args: argparse):
    return np.random.random(size=args.feature_size)


def get_valid_feature_matrix(facts: list,
                             args: argparse,
                             e_list: list,
                             r_list: list):
    valid_data_list = []
    valid_label_list = []
    # print(f"为每个fact生成fake 测试数据。")
    for triple in facts:
        feature_matrix_with_rid = torch.randn((args.entity_num, args.feature_size + 1))
        rid = r_list.index(triple[1])
        feature_matrix_with_rid[:, 0] = rid
        tail_label_idx = e_list.index(triple[2])
        tmp_label = torch.from_numpy(np.array(tail_label_idx))
        valid_data_list.append(feature_matrix_with_rid)
        valid_label_list.append(tmp_label)
    return valid_data_list, valid_label_list


def LoadValidGraph(args: argparse):
    facts_list, e_list, r_list = generate_fake_data(args, "valid")
    valid_feature_matrix_with_rid_and_labels = get_valid_feature_matrix(facts_list, args, e_list, r_list)
    return valid_feature_matrix_with_rid_and_labels


def LoadTestGraph(args: argparse):
    facts_list, e_list, r_list = generate_fake_data(args, "test")
    test_feature_matrix_with_rid_and_labels = get_valid_feature_matrix(facts_list, args, e_list, r_list)
    return test_feature_matrix_with_rid_and_labels


def my_valid_data_partition(valid_facts: list,
                            gpu_num_per_hold: int):
    data_len = len(valid_facts)
    all_batch_num = int(data_len / gpu_num_per_hold)
    batch_list = []
    for i in range(all_batch_num):
        batch_list.append(valid_facts[i*gpu_num_per_hold:(i+1)*gpu_num_per_hold])
    if data_len % gpu_num_per_hold != 0:
        batch_list.append(valid_facts[all_batch_num*gpu_num_per_hold:])
    return batch_list


def tail_predict(rank: int,
                 model_pt: MhYaoPRA,
                 valid_data: tuple,
                 args: argparse,
                 hold_out_id: int):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.gpu_num_per_hold,
                            rank=rank)
    gpu_id = hold_out_id * args.hold_out_num + rank
    print(f"hold_out_id/gpu_id/pid:[{hold_out_id}/{gpu_id}/{os.getpid()}].")
    torch.manual_seed(gpu_id)
    model_pt.cuda(gpu_id)
    model = torch.nn.parallel.DistributedDataParallel(model_pt,
                                                      device_ids=[gpu_id],
                                                      output_device=gpu_id)
    # 还在CPU上的三元组， 每个Torch进程都有一份指向它们的指针
    # 以Batch(大小为gpu_num_per_hold)形式遍历所有测试三元组,一张卡计算一条三元组
    # 每条三元组先由CPU生成对应的矩阵，然后传入GPU，并计算
    valid_facts, e_list, r_list = valid_data
    batch_list = my_valid_data_partition(valid_facts, args.gpu_num_per_hold)
    for batch_facts in tqdm(batch_list):
        # 调用CPU生成对应的特征矩阵list和标签list
        data, label = get_valid_feature_matrix(facts=batch_facts,
                                               args=args,
                                               e_list=e_list,
                                               r_list=r_list)
        valid_dataset = MhYaoPRAData(data=data,label=label)
        valid_sampler = torch.utils.data.DistributedSampler(dataset=valid_dataset,
                                                            num_replicas=args.gpu_num_per_hold,
                                                            rank=rank)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=int(args.batch_size / args.gpu_num_per_hold),
                                                   shuffle=False,
                                                   sampler=valid_sampler,
                                                   pin_memory=True,
                                                   num_workers=0)
        for feature_matrix_with_rid, label in valid_loader:
            feature_matrix_with_rid = feature_matrix_with_rid.cuda(gpu_id)
            results = model(feature_matrix_with_rid)


def train_this_hold_out(rank: int,
                        hold_out_id: int,
                        model_pt: MhYaoPRA,
                        args: argparse,
                        train_data):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.gpu_num_per_hold,
                            rank=rank)
    gpu_id = hold_out_id * args.hold_out_num + rank
    print(f"hold_out_id/gpu_id/pid:[{hold_out_id}/{gpu_id}/{os.getpid()}].")
    torch.manual_seed(gpu_id)
    model_pt.cuda(gpu_id)
    model = DDP(model_pt, device_ids=[gpu_id], output_device=gpu_id)
    feature_matrix_with_rid, labels = train_data
    train_data = MhYaoPRAData(data=feature_matrix_with_rid, label=labels)
    train_sampler = torch.utils.data.DistributedSampler(dataset=train_data,
                                                        num_replicas=args.gpu_num_per_hold,
                                                        rank=rank)
    valid_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=int(args.batch_size / args.gpu_num_per_hold),
                                               shuffle=False,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=0)
    criterion = nn.BCELoss().cuda(gpu_id)
    optimizer = torch.optim.Adam(model.parameters())
    loss = 0
    for epo in range(args.epochs):
        for feature_matrix_with_rid, label in valid_loader:
            feature_matrix_with_rid = feature_matrix_with_rid.cuda(gpu_id)
            label = label.cuda(gpu_id).squeeze()
            results = model(feature_matrix_with_rid)
            loss = criterion(results, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if hold_out_id == 0 and rank == 0:
            print(f"Step:[{epo + 1}/{args.epochs}];loss:{loss}.")


def one_hold_out(args: argparse,
                 hold_out_id: int):
    # 生成一个ground_truth模型
    true_model = MhYaoPRA(args.feature_size, args.relation_num)
    # 载入各种形式的数据  (matrix, labels)
    train_feature_matrix_with_rid_and_labels = LoadTrainGraph(args=args, model_pt=true_model)
    valid_data = generate_fake_data(args=args,train_valid_test="valid")

    # 开始训练
    param_list = [0.1, 0.2, 0.3]
    for param in param_list:
        print(f"开始搜索超参param:{param}。")
        # 不同TorchProcess间共享模型，共享训练数据。
        # 一定记得释放掉共享的内存，因为下一套param还要重新开辟内存
        pra_model = MhYaoPRA(args.feature_size, args.relation_num)
        # 开始valid测试过程，调用torch multiprocessing
        os.environ['MASTER_ADDR'] = '172.17.0.4'
        os.environ['MASTER_PORT'] = str(hold_out_id + 8887)
        print(f"开始训练数据。")
        mp.spawn(fn=train_this_hold_out,
                 args=(hold_out_id,
                       pra_model,
                       args,
                       train_feature_matrix_with_rid_and_labels),
                 nprocs=args.gpu_num_per_hold)
        print(f"开始测试数据。")
        mp.spawn(fn=tail_predict,
                 args=(pra_model,
                       valid_data,
                       args,
                       hold_out_id),
                 nprocs=args.gpu_num_per_hold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--entity_num', default=15000, type=int)
    parser.add_argument('-rn', '--relation_num', default=302, type=int)
    parser.add_argument('-trainfn', '--train_facts_num', default=50000, type=int)
    parser.add_argument('-validfn', '--valid_facts_num', default=52000, type=int)
    parser.add_argument('-testfn', '--test_facts_num', default=52000, type=int)
    parser.add_argument('-hon', '--hold_out_num', default=1, type=int)
    parser.add_argument('-gnph', '--gpu_num_per_hold', default=2, type=int)
    parser.add_argument('-bsz', '--batch_size', default=2, type=int)
    parser.add_argument('-fsz', '--feature_size', default=100, type=int)
    parser.add_argument('-psz', '--pos_sample_size', default=1000, type=int)
    parser.add_argument('-epo', '--epochs', default=100, type=int)
    args = parser.parse_args()
    # 模拟LanuchHoldOutTest.py中，为各个HoldOut打开MyPool进程
    hold_out_pool = MyPool(processes=args.hold_out_num)
    for hold_out_id in range(args.hold_out_num):
        hold_out_pool.apply_async(func=one_hold_out,
                                  args=(args,
                                        hold_out_id,),
                                  error_callback=handle_error)
    hold_out_pool.close()
    hold_out_pool.join()


if __name__ == "__main__":
    main()
