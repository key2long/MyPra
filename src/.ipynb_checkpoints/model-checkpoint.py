import torch
import torch.nn as nn
import pdb

class MhYaoPRA(nn.Module):
    """This model requires all relation are indexed. The first relation's classifier's parameter is the first row vec.

    Args:
        feature_size: int. The size of each relation's classifier's parameter.
        num_relation: int. Total number of relations.

    Attributes:
        classifiers: torch.tensor. Its shape is [relation_num, feature_size].
                                    Each row is the parameter of one relation's classifier's parameter.

    Methods:
        forward: The input tensor is of shape [1, batch_size, 1 + feature_size]. The first dimension "1" can be neglect.
                In training stage, the input tensor would be one relation's feature matrix.
                That is, all this relation's training triple (both pos and neg)'s number equals to batch_size.
                And, the first row of this feature matrix will be this relation's index.
                In valid/test stage, the input tensor would be one triple's feature matrix.
                That is, for one triple (h, r, t) in valid graph, by replacing the tail entity "t" with all
                possible entities, we can get set {(h, r, t_i)}_{i=1,...,|E|}. |E| denotes the number of all entities.
                Then, each triple (h, r, t_i) corresponds to one row of this feature matrix, and therefore batch_size=|E|.
                The feature_size dim remains unchanged.
    """
    def __init__(self,
                 feature_size: int,
                 relation_num: int):
        super(MhYaoPRA, self).__init__()
        classifiers = torch.rand((relation_num, feature_size), requires_grad=True, dtype=torch.float32)
        # w = torch.empty((relation_num, feature_size), requires_grad=True, dtype=torch.float32)
        # classifiers = torch.nn.init.kaiming_normal_(w)
        self.classifiers = torch.nn.Parameter(classifiers)
        self.sigmod = nn.Sigmoid()
        self.register_parameter("classifiers", self.classifiers)

    def forward(self,
                batch_features_with_rid: torch.tensor):
        """Forward function that works both in training stage and validation stage.

        :param batch_features_with_rid: (rid, batch_features_of_rid)
                                         where [batch_features_of_rid]_{1*batch_size*feature_size}
        :return: results
        """
        rid = int(batch_features_with_rid[0, 0, 0])
        # print(type(batch_features_with_rid[0, :, 1:]), type(self.classifiers[rid, :]))
        # pdb.set_trace()
        scores = torch.matmul(batch_features_with_rid[0, :, 1:], self.classifiers[rid, :])
        results = self.sigmod(scores)
        return results
