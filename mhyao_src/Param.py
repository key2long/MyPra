import yaml


class Param:

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.hyper_param = None
        if config_path is not None:
            self._loadHyperParam()

    def _loadHyperParam(self):
        with open(self.config_path) as config_file:
            self.hyper_param = yaml.load(config_file, Loader=yaml.FullLoader)

    def typeTrans(self, str_param, param_type):
        """
        目前只支持将字符串转化为python内置的类型:int, float
        :param str_param: 字符串
        :return:
        """
        if isinstance(str_param, str) is not True:
            raise Exception(f"变量{str_param}不是字符串.请输入字符串版本.")
        if param_type == "str":
            return str_param
        elif param_type == "int":
            try:
                return int(str_param)
            except ValueError:
                raise Exception(f"变量{str_param}无法转化为int型数据.")
        elif param_type == "float":
            try:
                return float(str_param)
            except ValueError:
                raise Exception(f"变量{str_param}无法转化为float型数据.")
        else:
            raise Exception("还不支持该类型转化")

    def ifInConfig(self, param_dict: dict, param_type: str = None):
        """
        查询param是否在yml文件中,如果param为空且yml文件中还有记录,那么就返回yml中的记录,否则原样返回
        假设某个参数的名字是param,具体值为1；
        则dict(param=param)返回的结果会是一个字典:{"param": 1}；
        上述字典便是param_dict的一个例子
        :param param_dict: {"name_of_param": param}
        :param param_type: 参数类型
        :return: param
        """
        name_of_param = list(param_dict.keys())[0]
        param = param_dict[name_of_param]
        if self.hyper_param is None:
            return param
        if param is not None:
            return param
        if param is None:
            if name_of_param in self.hyper_param:
                param_yml = self.hyper_param[name_of_param]
                # todo: 检查param_yml是否是param_type类型
                return param_yml
            else:
                return param
