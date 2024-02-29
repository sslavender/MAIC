## Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python tqdm imageio pandas matplotlib tensorboardX` 



## Dataset Preparation

### Download datasets

**CelebA** dataset can be downloaded [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Please download and unzip the `img_celeba.7z` file. 

**Helen** dataset can be downloaded [here](http://www.ifp.illinois.edu/~vuongle2/helen/). Please download and unzip the 5 parts of `All images`. 

**Testing sets** for CelebA and Helen can be downloaded from [Google Drive](https://drive.google.com/open?id=1Q1T1smMDRMO1NcjkxbZvotOX93YIVp5e) or [Baidu Drive](https://pan.baidu.com/s/14zJ_lY8iFmk3csHYZmut7Q) (extraction code: 6qhx). 

### Download landmark annotations and pretrained models

**Landmark annotations** for CelebA and Helen can be downloaded in the `annotations` folder from [Google Drive](https://drive.google.com/open?id=1Q1T1smMDRMO1NcjkxbZvotOX93YIVp5e) or [Baidu Drive](https://pan.baidu.com/s/14zJ_lY8iFmk3csHYZmut7Q) (extraction code: 6qhx). 

The **pretrained models** can also be downloaded from the `models` folder in the above links. Then please place them in `./models`. 
