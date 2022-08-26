# PolyNet_Pytorch
This repository contains the official code to reproduce the results from the paper: 

**PolyNet: Polynomial Neural Network for 3D Shape Recognition with PolyShape Representation (3DV 2021)**

\[[project page](https://myavartanoo.github.io/polynet/)\] \[[arXiv](https://arxiv.org/abs/2110.07882)\] \[[ResearchGate](https://www.researchgate.net/publication/355218072_PolyNet_Polynomial_Neural_Network_for_3D_Shape_Recognition_with_PolyShape_Representation)\]  \[[presentation](https://www.youtube.com/watch?v=Pk8gvfGV5N8&list=PLhCEMvtuQ92VkQCsiMKSn5ORcYM3BA1eM)\] 


<p align="center">
<img src="source/PolyNet.png" width="100%"/>  
</p>


### Dependencies
* Python 3.8.5
* PyTorch 1.7.1
* numpy
* Pillow
* torch_scatter

### Dataset
Download the preprocessed ModelNet dataset with PTQ and √3-subdivision from the follwing link and unzip them in the data directroy. The data type is ```.npz```.

\[[PTQ](https://drive.google.com/drive/folders/15VFhxRTpSfetJqNqssuNWNJhpHOBOUsE?usp=sharing)\] \[[√3-subdivision](https://drive.google.com/drive/folders/1WnwZ0NkSme9s_VceZRjVTpS1QYCJ4yYt?usp=sharing)\] 


### Train
In ```config.json``` you can set dataset type (ModelNet10 or ModelNet40) and the PolyPool type (PTQ, Sqrt3).

To train PolyNet with the desired dataset and PolyPool, simply run, 

```
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json -t "direction to save the model"
```



## Citation
If you find our paper, code, or provided data useful, please consider citing:

```
@INPROCEEDINGS{9665897,
  author={Yavartanoo, Mohsen and Hung, Shih-Hsuan and Neshatavar, Reyhaneh and Zhang, Yue and Lee, Kyoung Mu},
  booktitle={2021 International Conference on 3D Vision (3DV)}, 
  title={PolyNet: Polynomial Neural Network for 3D Shape Recognition with PolyShape Representation}, 
  year={2021},
  pages={1014-1023},
  doi={10.1109/3DV53792.2021.00109}}
```

