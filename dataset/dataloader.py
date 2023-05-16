from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module
    # 此行导入在args.dataset属性中指定的数据集的模块
    # 使用importlib库中的import_module（）函数
    m = import_module('dataset.' + args.dataset.lower())

    # 如果指定的数据集是RRSSRD
    if (args.dataset == 'RRSSRD'):
        # 使用导入模块中的TrainSet类创建TrainSet对象，并将args对象作为参数传递
        data_train = getattr(m, 'TrainSet')(args)
        # 创建一个DataLoader对象，使用torch.utils.data库中的DataLoader类加载train数据集
        # 并传递data_train、batch_size、shuffle和num_workers作为参数
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        # 创建一个空字典dataloader_test
        dataloader_test = {}
        for i in range(5):
            # 使用导入模块中的TestSet类创建TestSet对象
            # 并将args对象和ref_level作为参数传递
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            # 创建一个DataLoader对象，使用torch.utils.data库中的DataLoader类加载test数据集
            # 并传递data_test、batch_size、shuffle和num_workers作为参数
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        # 创建一个带有“train”和“test”键的字典dataloader
        # 以及分别作为dataloader_train和dataloader_test的值
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader