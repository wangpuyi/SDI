import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.base import get_counts

wnids = ['n01514668', 'n01530575', 'n01616318', 'n01629819', 'n01687978', 
         'n01695060', 'n01698640', 'n01729977', 'n01773157', 'n01773549',
         'n01784675', 'n01806143', 'n01824575', 'n01829413', 'n01860187',
         'n01872401', 'n01877812', 'n01883070', 'n01924916', 'n01943899', 
         'n01980166', 'n02002556', 'n02002724', 'n02017213', 'n02033041', 
         'n02085620', 'n02085782', 'n02086079', 'n02087046', 'n02087394', 
         'n02088364', 'n02093859', 'n02097130', 'n02097474', 'n02098105', 
         'n02098286', 'n02098413', 'n02102040', 'n02102177', 'n02105641', 
         'n02106166', 'n02106382', 'n02106550', 'n02107142', 'n02110185', 
         'n02110806', 'n02111277', 'n02111500', 'n02111889', 'n02113624', 
         'n02115913', 'n02116738', 'n02125311', 'n02128925', 'n02129165', 
         'n02134418', 'n02168699', 'n02174001', 'n02177972', 'n02259212', 
         'n02268443', 'n02281787', 'n02328150', 'n02342885', 'n02361337', 
         'n02395406', 'n02396427', 'n02410509', 'n02415577', 'n02417914', 
         'n02422699', 'n02486261', 'n02488291', 'n02488702', 'n02514041', 
         'n02687172', 'n02708093', 'n02782093', 'n02791124', 'n02815834', 
         'n02817516', 'n02837789', 'n02870880', 'n02906734', 'n02910353', 
         'n02927161', 'n02948072', 'n03014705', 'n03026506', 'n03028079', 
         'n03065424', 'n03075370', 'n03095699', 'n03110669', 'n03124043', 
         'n03160309', 'n03188531', 'n03201208', 'n03250847', 'n03255030', 
         'n03272010', 'n03344393', 'n03347037', 'n03388043', 'n03393912', 
         'n03417042', 'n03424325', 'n03444034', 'n03445777', 'n03461385', 
         'n03485407', 'n03498962', 'n03529860', 'n03535780', 'n03584829', 
         'n03595614', 'n03599486', 'n03637318', 'n03662601', 'n03666591', 
         'n03697007', 'n03706229', 'n03709823', 'n03721384', 'n03743016', 
         'n03769881', 'n03775071', 'n03782006', 'n03785016', 'n03832673', 
         'n03841143', 'n03873416', 'n03891332', 'n03895866', 'n03903868', 
         'n03908618', 'n03944341', 'n03950228', 'n03954731', 'n03961711', 
         'n03970156', 'n03976467', 'n03982430', 'n04009552', 'n04049303', 
         'n04067472', 'n04152593', 'n04154565', 'n04162706', 'n04264628', 
         'n04265275', 'n04266014', 'n04311004', 'n04311174', 'n04328186', 
         'n04330267', 'n04344873', 'n04347754', 'n04357314', 'n04367480', 
         'n04371774', 'n04399382', 'n04417672', 'n04428191', 'n04435653', 
         'n04443257', 'n04479046', 'n04483307', 'n04486054', 'n04501370', 
         'n04507155', 'n04548362', 'n04552348', 'n04554684', 'n04557648', 
         'n04562935', 'n04579145', 'n04590129', 'n04604644', 'n04606251', 
         'n06785654', 'n06874185', 'n07565083', 'n07583066', 'n07615774', 
         'n07695742', 'n07715103', 'n07716906', 'n07718472', 'n07734744', 
         'n07742313', 'n07745940', 'n07860988', 'n07932039', 'n09288635', 
         'n09399592', 'n09472597', 'n13037406', 'n13044778', 'n13054560']

class FilteredImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        # 保存指定要保留的文件夹名称
        self.included_folders = wnids
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        # 重写 find_classes 方法，只包含指定的类
        all_classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        if self.included_folders:
            classes = [cls for cls in all_classes if cls in self.included_folders]
        else:
            classes = all_classes
        classes.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        # 重写 make_dataset 方法，过滤掉不在 included_folders 中的文件夹
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_to_idx.keys()):
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            # 如果该文件夹在 included_folders 中，则处理
            if target_class in self.included_folders:
                for root, _, fnames in sorted(os.walk(target_dir)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if is_valid_file is None or is_valid_file(path):
                            item = (path, class_to_idx[target_class])
                            instances.append(item)
        return instances

class IN200_sub_generalize(FilteredImageFolder):
    def __init__(self, root, transform=None, group=0, cfg=None):
        self.group = group
        super().__init__(root, transform=transform)
        self.groups = [0]*len(self.samples)
        self.group_names = ['all']
        # self.split = split

        self.class_names = self.classes # n03733281
        self.class_map = None
        self.targets = [s[1] for s in self.samples]

        self.class_weights = get_counts(self.targets)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, 0
