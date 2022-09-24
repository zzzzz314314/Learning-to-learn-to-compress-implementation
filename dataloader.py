import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


class JPEG_AI(Dataset):

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set (task)
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.RandomCrop((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])
            self.img_path = os.path.join(root, 'training_set')  # image path
            
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.CenterCrop((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 ])
            self.img_path = os.path.join(root, 'validation_set')  # image path
    
   
        #self.img_path = os.path.join(root, 'training_set')  # image path
        
        img_list = os.listdir(self.img_path)

        self.support_x_batch = img_list[:self.batchsz]
        # self.query_x_batch = img_list[:self.batchsz]
        # self.support_y_batch = img_list[:self.batchsz]
        # self.query_y_batch = img_list[:self.batchsz]
        
 
    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """

    
        # support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # support_y = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # query_y = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)


        support_x = self.transform(os.path.join(self.img_path, self.support_x_batch[index]))
        # query_x = self.transform(os.path.join(self.img_path, self.query_x_batch[index]))
        # support_y = self.transform(os.path.join(self.img_path, self.support_y_batch[index]))
        # query_y = self.transform(os.path.join(self.img_path, self.query_y_batch[index]))

        return support_x #, support_y, query_x, query_y

    def __len__(self):

        return self.batchsz


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    #tb = SummaryWriter('runs', 'imagenet64')
    jpeg_ai = JPEG_AI('jpeg_ai', mode='train', n_way=1, k_shot=1, k_query=1, batchsz=5, resize=256) 

    for i, set_ in enumerate(jpeg_ai):

        # support_x, support_y, query_x, query_y = set_
        support_x = set_
        # print(support_x)

        support_x = make_grid(support_x, nrow=2)
        # query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        # plt.figure(2)
        # plt.imshow(query_x.transpose(2, 0).numpy())
        # plt.pause(0.5)

        # tb.add_image('support_x', support_x)
        # tb.add_image('query_x', query_x)

        #time.sleep(5)

    # tb.close()
