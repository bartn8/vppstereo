
<h1 align="center"> Active Stereo Without Pattern Projector (ICCV 2023) </h1> 
<h1 align="center"> Stereo-Depth Fusion through Virtual Pattern Projection (Journal Extension) </h1> 

<br>

:rotating_light: This repository contains download links to our code, and trained deep stereo models of our works  "**Active Stereo Without Pattern Projector**",  [ICCV 2023](https://iccv2023.thecvf.com/) and "**Stereo-Depth Fusion through Virtual Pattern Projection**", Journal Extension
 
by [Luca Bartolomei](https://bartn8.github.io/)<sup>1,2</sup>, [Matteo Poggi](https://mattpoggi.github.io/)<sup>2</sup>, [Fabio Tosi](https://fabiotosi92.github.io/)<sup>2</sup>, [Andrea Conti](https://andreaconti.github.io/)<sup>2</sup>, and [Stefano Mattoccia](https://github.com/stefano-mattoccia)<sup>1,2</sup>

Advanced Research Center on Electronic System (ARCES)<sup>1</sup>
University of Bologna<sup>2</sup>

<div class="alert alert-info">

<h2 align="center"> 

 Active Stereo Without Pattern Projector (ICCV 2023)<br>

 [Project Page](https://vppstereo.github.io/) | [Paper](https://vppstereo.github.io/assets/paper.pdf) |  [Supplementary](https://vppstereo.github.io/assets/paper-supp.pdf) | [Poster](https://vppstereo.github.io/assets/poster.pdf)
</h2>

**Note**: 🚧 Kindly note that this repository is currently in the development phase. We are actively working to add and refine features and documentation. We apologize for any inconvenience caused by incomplete or missing elements and appreciate your patience as we work towards completion.

## :bookmark_tabs: Table of Contents

- [:bookmark\_tabs: Table of Contents](#bookmark_tabs-table-of-contents)
- [:clapper: Introduction](#clapper-introduction)
- [:movie\_camera: Watch Our Research Video!](#movie_camera-watch-our-research-video)
- [:inbox\_tray: Pretrained Models](#inbox_tray-pretrained-models)
- [:memo: Code](#memo-code)
  - [:hammer\_and\_wrench: Setup Instructions](#hammer_and_wrench-setup-instructions)
- [:floppy_disk: Datasets](#floppy_disk-datasets)
- [:rocket: Test](#rocket-test)
- [:art: Qualitative Results](#art-qualitative-results)
- [:envelope: Contacts](#envelope-contacts)
- [:pray: Acknowledgements](#pray-acknowledgements)

</div>

## :clapper: Introduction
This paper proposes a novel framework integrating the principles of active stereo in standard passive camera systems without a physical pattern projector.
We virtually project a pattern over the left and right images according to the sparse measurements obtained from a depth sensor.


<h4 align="center">

</h4>

<img src="./images/framework.jpg" alt="Alt text" style="width: 800px;" title="architecture">


**Contributions:** 

* Even with meager amounts of sparse depth seeds (e.g., 1% of the whole image), our approach outperforms by a large margin state-of-the-art sensor fusion methods based on handcrafted algorithms and deep networks.

* When dealing with deep networks trained on synthetic data, it dramatically improves accuracy and shows a compelling ability to tackle domain shift issues, even without additional training or fine-tuning.

* By neglecting a physical pattern projector, our solution works under sunlight, both indoors and outdoors, at long and close ranges with no additional processing cost for the original stereo matcher.

:fountain_pen: If you find this code useful in your research, please cite:

```bibtex
@InProceedings{Bartolomei_2023_ICCV,
    author    = {Bartolomei, Luca and Poggi, Matteo and Tosi, Fabio and Conti, Andrea and Mattoccia, Stefano},
    title     = {Active Stereo Without Pattern Projector},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18470-18482}
}
```

## :movie_camera: Watch Our Research Video!

<a href="https://vppstereo.github.io/#myvideo">
  <img src="./images/slide_title.jpg" alt="Watch the video" width="800">
</a>



## :inbox_tray: Pretrained Models

Here, you can download the weights of **RAFT-Stereo** and **PSMNet** architectures. 
- **Vanilla Models**: these models are pretrained on Sceneflow vanilla images and Middlebury vanilla images
  - PSMNet vanilla models: _psmnet/sceneflow/psmnet.tar_, _psmnet/middlebury/psmnet.tar_
  - RAFT-Stereo vanilla models (_raft-stereo/sceneflow/raftstereo.pth_ and _raft-stereo/middlebury/raftstereo.pth_) are just a copy from [authors' drive](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J)
- **Fine-tuned Models**: starting from vanilla models, these models (_*-vpp-ft.tar_) are finetuned in the same domain but with virtual projected images
- **Models trained from scratch**: these models (_*-vpp-tr.tar_) are trained from scratch using virtual projected images


To use these weights, please follow these steps:

1. Install [GDown](https://github.com/wkentaro/gdown) python package: `pip install gdown`
2. Download all weights from our drive: `gdown --folder https://drive.google.com/drive/folders/1GqcY-Z-gtWHqDVMx-31uxrPzprM38UJl?usp=drive_link`

## :memo: Code

The **Test** section provides scripts to evaluate disparity estimation models on datasets like **KITTI**, **Middlebury**, and **ETH3D**. It helps assess the accuracy of the models and saves predicted disparity maps.

Please refer to each section for detailed instructions on setup and execution.

<div class="alert alert-info">

**Warning**:
- Please be aware that we will not be releasing the training code for deep stereo models. The provided code focuses on evaluation and demonstration purposes only. 
- With the latest updates in PyTorch, slight variations in the quantitative results compared to the numbers reported in the paper may occur.

</div>


### :hammer_and_wrench: Setup Instructions

1. **Dependencies**: Ensure that you have installed all the necessary dependencies. The list of dependencies can be found in the `./requirements.txt` file.
2. **Build rSGM**: 
  - Firstly, please initialize and update **git submodules**: `git submodule init; git submodule update`
  - Go to `./thirdparty/stereo-vision/reconstruction/base/rSGM/`
  - Build and install pyrSGM package: `python setup.py build_ext --inplace install`


## :floppy_disk: Datasets
We used seven datasets for training and evaluation.

### Middlebury

**Midd-14**: We used the [MiddEval3](https://vision.middlebury.edu/stereo/eval3/) training split for evaluation and fine-tuning purposes.

```bash
$ cd PATH_TO_DOWNLOAD
$ wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip
$ wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-F.zip
$ unzip \*.zip
```

After that, you will get a data structure as follows:

```
MiddEval3
├── TrainingF
│    ├── Adirondack
│    │    ├── im0.png
│    │    └── ...
|    ...
|    └── Vintage
│         └── ...
└── TestF
     └── ...
```


**Midd-A**: We used the [Scenes2014](https://vision.middlebury.edu/stereo/data/scenes2014/) additional split for evaluation and grid-search purposes.

```bash
$ cd PATH_TO_DOWNLOAD
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Backpack-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Bicycle1-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Cable-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Classroom1-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Couch-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Flowers-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Mask-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Shopvac-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sticks-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Storage-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword1-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword2-perfect.zip
$ wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Umbrella-perfect.zip
$ unzip \*.zip
```

After that, you will get a data structure as follows:

```
middlebury2014
├── Backpack-perfect
│    ├── im0.png
│    └── ...
...
└── Umbrella-perfect
     └── ...
```

**Midd-21**: We used the [Scenes2021](https://vision.middlebury.edu/stereo/data/scenes2021/) split for evaluation purposes.

```bash
$ cd PATH_TO_DOWNLOAD
$ wget https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip
$ unzip all.zip
$ mv data/* .
```

After that, you will get a data structure as follows:

```
middlebury2021
├── artroom1
│    ├── im0.png
│    └── ...
...
└── traproom2
     └── ...
```

Note that additional datasets are available at the [official website](https://vision.middlebury.edu/stereo/data/).

### KITTI142

We based our KITTI142 validation split from [KITTI141](https://github.com/XuelianCheng/LidarStereoNet) (we added frame 000124). You can download it from our drive using this script:

```bash
$ cd PATH_TO_DOWNLOAD
$ gdown --fuzzy https://drive.google.com/file/d/1A14EMqcGLDhH3nTHTVFpSP2P7We0SY-C/view?usp=drive_link
$ unzip kitti142.zip
```

After that, you will get a data structure as follows:

```
kitti142
├── image_2
│    ├── 000002_10.png
|    ...
│    └── 000199_10.png
├── image_3
│    ├── 000002_10.png
|    ...
│    └── 000199_10.png
├── lidar_disp_2
│    ├── 000002_10.png
|    ...
│    └── 000199_10.png
├── disp_occ
│    ├── 000002_10.png
|    ...
│    └── 000199_10.png
...
```

Note that additional information are available at the [official website](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

### ETH3D

You can download [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data) dataset following this script:

```bash
$ cd PATH_TO_DOWNLOAD
$ wget https://www.eth3d.net/data/two_view_training.7z
$ wget https://www.eth3d.net/data/two_view_training_gt.7z
$ p7zip -d *.7z
```

After that, you will get a data structure as follows:

```
eth3d
├── delivery_area_1l
│    ├── im0.png
│    └── ...
...
└── terrains_2s
     └── ...
```
Note that the script erases 7z files. Further details are available at the [official website](https://www.eth3d.net/datasets).

### DSEC

We provide preprocessed DSEC testing splits _Day_, _Afternoon_ and _Night_:

```bash
$ cd PATH_TO_DOWNLOAD
$ gdown --folder https://drive.google.com/drive/folders/1etkvdntDfMdwvx_NP0_QJcUcsogLXYK7?usp=drive_link
$ cd dsec
$ unzip -o \*.zip
$ cd ..
$ mv dsec/* .
$ rmdir dsec
```

After that, you will get a data structure as follows:

```
dsec
├── afternoon
│    ├── left
|    |    ├── 000000.png
|    |    ...
│    └── ...
...
└── night
     └── ...
```

We managed to extract the splits using only data from the [official website](https://dsec.ifi.uzh.ch/dsec-datasets/download/).
We used [FasterLIO](https://github.com/gaoxiang12/faster-lio) to de-skew raw LiDAR scans and [Open3D](https://github.com/isl-org/Open3D) to perform ICP registration.

### M3ED

We provide preprocessed M3ED testing splits _Outdoor Day_, _Outdoor Night_ and _Indoor_:


```bash
$ cd PATH_TO_DOWNLOAD
$ gdown --folder https://drive.google.com/drive/folders/1n-7H11ZfbPcR9_F0Ri2CcTJS2WWQlfCo?usp=drive_link
$ cd m3ed
$ unzip -o \*.zip
$ cd ..
$ mv m3ed/* .
$ rmdir m3ed
```

After that, you will get a data structure as follows:

```
m3ed
├── indoor
│    ├── left
|    |    ├── 000000.png
|    |    ...
│    └── ...
...
└── night
     └── ...
```

We managed to extract the splits using only data from the [official website](https://github.com/daniilidis-group/m3ed).

### M3ED Active

We provide preprocessed M3ED Active testing splits _Passive_, and _Active_:


```bash
$ cd PATH_TO_DOWNLOAD
$ gdown --folder https://drive.google.com/drive/folders/1fv6f2mQUPW8MwSsGy1f0dEHOZCS4sk2-?usp=drive_link
$ cd m3ed_active
$ unzip -o \*.zip
$ cd ..
$ mv m3ed_active/* .
$ rmdir m3ed_active
```

After that, you will get a data structure as follows:

```
m3ed_active
├── passive
│    ├── left
|    |    ├── 000000.png
|    |    ...
│    └── ...
└── active
     └── ...
```

We managed to extract the splits using only data from the [official website](https://github.com/daniilidis-group/m3ed).

### SIMSTEREO

You can download [SIMSTEREO](https://arxiv.org/abs/2209.08305) dataset [here](https://ieee-dataport.org/open-access/active-passive-simstereo).

After that, you will get a data structure as follows:

```
simstereo
├── test
│    ├── nirColormanaged
|    |    ├── abstract_bowls_1_left.jpg
|    |    ├── abstract_bowls_1_right.jpg
|    |    ...
│    ├── rgbColormanaged
|    |    ├── abstract_bowls_1_left.jpg
|    |    ├── abstract_bowls_1_right.jpg
|    |    ...
│    └── pfmDisp
|         ├── abstract_bowls_1_left.pfm
|         ├── abstract_bowls_1_right.pfm
|         ...
└── training
     └── ...
```


## :rocket: Test

This code snippet allows you to evaluate the disparity maps on various datasets, including [KITTI (142 split)](https://www.cvlibs.net/datasets/kitti/o), [Middlebury (Training, Additional, 2021)](https://vision.middlebury.edu/stereo/data/), [ETH3D](https://www.eth3d.net/), [DSEC](https://dsec.ifi.uzh.ch/), [M3ED](https://m3ed.io/), and [SIMSTEREO](https://ieee-dataport.org/open-access/active-passive-simstereo). By executing the provided script, you can assess the accuracy of disparity estimation models on these datasets.

To run the `test.py` script with the correct arguments, follow the instructions below:

1. **Run the test**:
   - Open a terminal or command prompt.
   - Navigate to the directory containing the `test.py` script.

2. **Execute the command**:
   Run the following command, replacing the placeholders with the actual values for your images and model:

   ```shell
   # Parameters to reproduce Active Stereo Without Pattern Projector (ICCV 2023)
   CUDA_VISIBLE_DEVICES=0 python test.py  --datapath <path_to_dataset> --dataset <dataset_type> --stereomodel <model_name> \
    --loadstereomodel <path_to_pretrained_model> --maxdisp 192 \
    --vpp --outdir <save_dmap_dir> --wsize 3 --guideperc 0.05 --blending 0.4 --iscale <input_image_scale> \
    --maskocc
   ```

   ```shell
   # Parameters to reproduce Stereo-Depth Fusion through Virtual Pattern Projection (Journal Extension)
   CUDA_VISIBLE_DEVICES=0 python test.py  --datapath <path_to_dataset> --dataset <dataset_type> --stereomodel <model_name> \
    --loadstereomodel <path_to_pretrained_model> --maxdisp 192 \
    --vpp --outdir <save_dmap_dir> --wsize 7 --guideperc 0.05 --blending 0.4 --iscale <input_image_scale> \
    --maskocc --bilateralpatch --bilateral_spatial_variance 1 --bilateral_color_variance 2 --bilateral_threshold 0.001 --rsgm_subpixel
   ```

  Replace the placeholders (<max_disparity>, <path_to_dataset>, <dataset_type>, etc.) with the actual values for your setup.

  The available arguments are:

  - `--maxdisp`: Maximum disparity range for PSMNet and rSGM (default 192).
  - `--stereomodel`: Stereo model type. Options: `raft-stereo`, `psmnet`, `rsgm`
  - `--normalize`: Normalize RAFT-Stereo input between [-1,1] instead of [0,1] (Only for official weights)
  - `--datapath`: Specify the dataset path.
  - `--dataset`: Specify dataset type. Options: `kitti_stereo142`, `middlebury_add`, `middlebury2021`, `middlebury`, `eth3d`, `simstereo`, `simstereoir`, `dsec`, `m3ed`
  - `--outdir`: Output directory to save the disparity maps.
  - `--loadstereomodel`: Path to the pretrained model file.
  - `--iscale` Rescale input images before apply vpp and stereo matching. Original size is restored before evaluation. Example: `--iscale 1` equals full scale, `--iscale 2` equals half scale.
  - `--guideperc`: Simulate depth seeds using a certain percentage of randomly sampled GT points. Valid only if raw depth seeds do  not exists.
  - `--vpp`: Apply virtual patterns to stereo images
  - `--colormethod`: Virtual pattering strategy. Options: `rnd` (i.e., random strategy) and `maxDistance` (i.e., histogram based strategy)
  - `--uniform_color`: Uniform patch strategy
  - `--wsize`: Pattern patch size (e.g., 1, 3, 5, 7, ...)
  - `--wsizeAgg_x`: Histogram based search window width
  - `--wsizeAgg_y`: Histogram based search window height
  - `--blending`: Alpha-bleding between original images and virtual pattern
  - `--maskocc`: Use proposed occlusion handling
  - `--discard_occ`: Use occlusion point discard strategy
  - `--guided`: Apply Guided Stereo Matching strategy
  - `--bilateralpatch`: Use adaptive patch based on bilateral filter
  - `--bilateral_spatial_variance`: Spatial variance of the adaptive patch
  - `--bilateral_color_variance`: Color variance of the adaptive patch 
  - `--bilateral_threshold`: Adaptive patch classification threshold

For more details, please refer to the `test.py` script.

## :art: Qualitative Results

In this section, we present illustrative examples that demonstrate the effectiveness of our proposal.

<br>

<p float="left">
  <img src="./images/competitors.png" width="800" />
</p>
 
**Performance against competitors.** We can notice that VPP generally reaches almost optimal performance with a meagre 1% density and, except few cases in the -tr configurations with some higher density, achieves much lower error rates.
 
<br>

<p float="left">
  <img src="./images/vpp_ots.png" width="800" />
</p>

**VPP with off-the-shelf networks.** We collects the results yielded VPP applied to several off-the-shelf stereo models, by running
the weights provided by the authors. Again, VPP sensibly boosts the accuracy of any model with rare exceptions, either trained on synthetic or real data.

<br>

<p float="left">
  <img src="./images/teaser.png" width="800" />
</p>

**Qualitative Comparison on KITTI (top) and Middlebury (bottom).** From left to right: vanilla left images and disparity maps by PSMNet model, left images enhanced by our virtual projection and disparity maps by vanilla PSMNet model and (most right) vpp fine tuned PSMNet model.


<br>

<p float="left">
  <img src="./images/fine_details.png" width="800" />
</p>

**Fine-details preservation**: We can appreciate how our virtual pattern can greatly enhance the quality of the disparity maps, without introducing relevant artefacts in correspondence of thin structures – despite applying the pattern on patches.


## :envelope: Contacts

For questions, please send an email to luca.bartolomei5@unibo.it

## :pray: Acknowledgements

We would like to extend our sincere appreciation to the authors of the following projects for making their code available, which we have utilized in our work:

- We would like to thank the authors of [PSMNet](https://github.com/JiaRenChang/PSMNet), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo),[rSGM](https://github.com/ivankreso/stereo-vision) for providing their code, which has been instrumental in our stereo matching experiments.

We deeply appreciate the authors of the competing research papers for provision of code and model weights, which greatly aided accurate comparisons.

<h5 align="center">Patent pending - University of Bologna</h5>
