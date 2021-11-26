from __future__ import print_function, division
import os
from time import time
import numpy as np
import torch
from torch.utils.data import Dataset


time_print = False

class Shape3DDataset(Dataset):
    """Dataset for 3D shape generation"""

    def __init__(self, data_dir, task, dataset, PolyPool, transform=None):
        """
        csv_file: location of metadata.csv file
        data_dir: path where the data located
        point_limit: list of limit #points - on, out, in (in order)
        transform: Ignore this
        """
        self.dataset, self.PolyPool = dataset, PolyPool
        if dataset=="ModelNet10":
           self.cls_num = 10
           self.cls_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        elif dataset=="ModelNet40":
           self.cls_num = 40
           self.cls_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
                   
        self.cls_sizes = np.zeros([2,self.cls_num],dtype=int)
        self.task = task
        self.data_dir = data_dir
        if self.task == "train":
            if dataset=="ModelNet10":
               self.cls_sizes[1] = [106, 515, 889, 200, 200, 465, 200, 680, 392, 344]               
            elif dataset=="ModelNet40":
               self.cls_sizes[1] = [626,106,515,173,572,335,64,197,889,167,79,138,200,109,200,149,171,155,145,124,149,284,465,200,88,231,240,104,115,128,680,124,90,392,163,344,267,475,87,103]
        elif self.task == "test":
            if dataset=="ModelNet10":
               self.cls_sizes[0] = [106, 515, 889, 200, 200, 465, 200, 680, 392, 344]
               self.cls_sizes[1] = self.cls_sizes[0]+[50, 100, 100, 86, 86, 100, 86, 100, 100, 100]
            elif dataset=="ModelNet40":
               self.cls_sizes[0] = [626,106,515,173,572,335,64,197,889,167,79,138,200,109,200,149,171,155,145,124,149,284,465,200,88,231,240,104,115,128,680,124,90,392,163,344,267,475,87,103]
               self.cls_sizes[1] = self.cls_sizes[0]+[100,50,100,20,100,100,20,100,100,20,20,20,86,20,86,20,100,100,20,20,20,100,100,86,20,100,100,20,100,20,100,20,20,100,20,100,100,100,20,20]  
            

       
        self.cls_ind = np.arange(self.cls_num)



        self.indices = self._indices_generator()
        self.transform = transform



    def __len__(self):
        return np.sum(self.cls_sizes[1,:]-self.cls_sizes[0,:])

    def __getitem__(self, index):

        def _shape_loader(class_name, number_in_class):
            points = {}
            if self.PolyPool == "Sqrt3":
               if self.task=="train":
                  maximum = [44441,264015,83139,25103,5898]
               if self.task=="test":
                  maximum = [20300,123444,41148,13716,4572]
               path = self.data_dir + 'preprocessing/' + class_name + '/'+self.task+'/' +class_name +'_'+ str(number_in_class+1).zfill(4) + '.ply.npz'  
                  
            elif self.PolyPool == "PTQ":
               if self.task=="train":
                  maximum = [73075,590625,134853,28071,5898]
               if self.task=="test":
                  maximum = [48494,292608,73152,18288,4572]  
               path = self.data_dir + 'preprocessing_ptq/' + class_name + '/'+self.task+'/' +class_name +'_'+ str(number_in_class+1).zfill(4) + '.ply.npz'                  
                  

            shape3D = np.load(path)
            ver_num = np.zeros([2,4],dtype=int)
            ver_num[0,:] = shape3D['ver_num'].astype(np.int64)
            ver_num[1, :] = [shape3D['conv1'].shape[0],shape3D['conv2'].shape[0],shape3D['conv3'].shape[0],shape3D['conv4'].shape[0]]

            Input = np.zeros([maximum[0],6])
            Input[:ver_num[0,0],:6] = shape3D['input'][:ver_num[0,0],:6].reshape([-1, 6])

            adj1 = np.zeros([maximum[1]],dtype=int)
            adj2 = np.zeros([maximum[2]],dtype=int)
            adj3 = np.zeros([maximum[3]],dtype=int)
            adj4 = np.zeros([maximum[4]],dtype=int)
            adj1[:ver_num[1,0]] = shape3D['conv1']
            adj2[:ver_num[1,1]] = shape3D['conv2']
            adj3[:ver_num[1,2]] = shape3D['conv3']
            adj4[:ver_num[1,3]] = shape3D['conv4']



            c1 = np.zeros([maximum[1]],dtype=int)
            c2 = np.zeros([maximum[2]],dtype=int)
            c3 = np.zeros([maximum[3]],dtype=int)
            c4 = np.zeros([maximum[4]],dtype=int)
            c1[:ver_num[1,0]] = shape3D['ind1']
            c2[:ver_num[1,1]] = shape3D['ind2']
            c3[:ver_num[1,2]] = shape3D['ind3']
            c4[:ver_num[1,3]] = shape3D['ind4']





            return Input, adj1, adj2, adj3, adj4, c1, c2, c3, c4, ver_num



        def _timeprint(isprint, name, prevtime):
            if isprint:
                print('loading {} takes {} secs'.format(name, time()-prevtime))
            return time()

        if torch.is_tensor(index):
            index = index.tolist()

        # select class, #data
        number_in_class = self.indices[index,0]
        class_name = self.cls_names[self.indices[index,1]]
        class_ind = self.indices[index,1]
        # loading shapes
        Input, adj1, adj2, adj3, adj4, c1, c2, c3, c4, ver_num = _shape_loader(class_name, number_in_class)


        # loading projected images (silhouette)

        directory = np.array([class_ind, number_in_class])
        target = {'input': Input,
                  'adj1': adj1,
                  'adj2': adj2,
                  'adj3': adj3,
                  'adj4': adj4,
                  'c1': c1,
                  'c2': c2,
                  'c3': c3,
                  'c4': c4,
                  'ver_num': ver_num,
                  'class_num': class_ind,
                  'dir': directory # list, not Tensor
                  }

        # if self.transform:
        #     sample = self.transform(sample)

        return target, directory

    def _indices_generator(self):
        dif = self.cls_sizes[1,:]-self.cls_sizes[0,:]
        indices = np.zeros([sum(dif), 2])
        c = 0
        for ind in range(len(self.cls_sizes.T)):
            indices[c:dif[ind] + c,0] = np.arange(self.cls_sizes[0,ind],self.cls_sizes[1,ind])
            indices[c:dif[ind] + c,1] = self.cls_ind[ind]
            c = c + dif[ind]
        return indices.astype(int)






if __name__=="__main__":
    time_print=True

    prev = time()
    dataset = Shape3DDataset('../data/','train')
    print(dataset)
    max__ = [0,0,0,0,0]
    for i in range(0,10000):
         target = dataset[i]
         max_ = target[0]['max_']
         if max_[0]>max__[0]:
            max__[0]=max_[0]            
         if max_[1]>max__[1]:
            max__[1]=max_[1]     
         if max_[2]>max__[2]:
            max__[2]=max_[2]     
         if max_[3]>max__[3]:
            max__[3]=max_[3]     
         if max_[4]>max__[4]:
            max__[4]=max_[4]   
         print(i,max__)                                      
    #for i in target:
    #    print('{} shape: {}'.format(i,target[i].shape))
        
    print("Dataset setting - total {} seconds".format(time()-prev))



    
