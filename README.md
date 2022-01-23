# ulcer_classification-Title

<img src="imgs/overview.png" width="1200px"/>

### [Paper]() |  [Dataset]() | [Pretrained Models]() | [WebSite](https://deepmia.boun.edu.tr/) 

In this work, we proposed .....

## Example Results

### Grad-CAM++ with Real Images
<img src="imgs/GradCamFigure.png" width="1200px"/>

### Real and Synthetic Images
<img src="imgs/GradeDataVis (2).png" width="1200px"/>

## Dataset

The dataset used in this study is derived from the original [Hyper-Kvasir](https://datasets.simula.no/hyper-kvasir/) open-source dataset. Ulcerative Collitis(UC) images were classified according to the Mayo scoring method, which consists of score 0 representing no disease, score 1 representing mild disease, score 2 representing moderate disease, and score 3 representing severe disease.

We introduce new labelled 679 UC endoscopy images from [Hyper-Kvasir](https://datasets.simula.no/hyper-kvasir/) dataset including 128 for grade 0, 211 for grade 1, 217 for grade 2, and 123 for grade 3 were labelled by our experienced gastroenterologists.

The dataset used in this study includes a total of 509 UC endoscopy-labelled images with grade 0, grade 1, grade 2, and grade 3 from original [Hyper-Kvasir](https://datasets.simula.no/hyper-kvasir/) .

Our study contains a total of 1188 images from the UC endoscopy images;
- 150 grade 0 
- 351 grade 1 
- 456 grade 2 
- 219 grade 3




## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


### Getting started

- Clone this repo:
```bash
git clone https://github.com/DeepMIALab/Ulcer_grade_classificaiton
cd Ulcer_grade_classificaiton
```

- Install PyTorch 3.7 and other dependencies (e.g., torchvision, visdom, dominate, gputil).

- For pip users, please type the command `pip install -r requirements.txt`.

- For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

### Training and Test

- The image identity numbers which were used in train, validation and test sets are given as .txt files in [docs/](https://github.com/DeepMIALab/AI-FFPE/tree/main/docs) for both Brain and Lung dataset. To replicate the results, you may download [dataset]() and create a subset using these .txt files.

The data used for training are expected to be organized as follows:
```bash
DATASET                
 ├──  train
 |      ├──Grade_0
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png
 |      ├──Grade_1
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png
 |      ├──Grade_2
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png
 |      ├──Grade_3
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png
 ├──  test
 |      ├──Grade_0
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png
 |      ├──Grade_1
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png
 |      ├──Grade_2
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png
 |      ├──Grade_3
 |           ├── 1.png     
 |           ├── ...
 |           └── n.png

```

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the UC model:
```bash
python classifer.py --action train --train_data ./dataset/train --test_data ./dataset/test --model_name $MODEL --epoch_number $EPOCH_NUMBER
```

- Test the UC model:
```bash
python classifer.py --action test --train_data ./dataset/train --test_data ./dataset/test --model_name $MODEL --epoch_number $EPOCH_NUMBER
```

The test results will be saved to a html file here: ``` ./results/${result_dir_name}/latest_train/index.html ``` 


### Apply a pre-trained UC Grade Classificaiton model and evaluate
For reproducability, you can download the pretrained models for each algorithm [here.]()

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```
@misc{ozyoruk2021deep,
      title={Deep Learning-based Frozen Section to FFPE Translation}, 
      author={Kutsev Bengisu Ozyoruk and Sermet Can and Guliz Irem Gokceler and Kayhan Basak and Derya Demir and Gurdeniz Serin and Uguray Payam Hacisalihoglu and Berkan Darbaz and Ming Y. Lu and Tiffany Y. Chen and Drew F. K. Williamson and Funda Yilmaz and Faisal Mahmood and Mehmet Turan},
      year={2021},
      eprint={2107.11786},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```



### Acknowledgments
Our code is developed based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/tree/54a6cca27a9a3e092a07457f5d56709da56e3cf5). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation, and [FastGAN-pytorch](https://github.com/odegeasslbc/FastGAN-pytorch) for the PyTorch implementation of FastGAN used in our single-image translation setting.
