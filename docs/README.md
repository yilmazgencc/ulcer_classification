# SCI-AIDE : High-fidelity Few-shot Histopathology Image Synthesis for Rare Cancer Diagnosis
<p align="center">
 <img src="imgs/main_figure.jpeg" width="800px"/>
</p>

###  [Pretrained Models]() | [Website](https://deepmia.boun.edu.tr/) 

In this work, we created synthetic tissue microscopy images using few-shot learning and developed a digital pathology pipeline called SCI-AIDE to improve diagnostic accuracy. Since rare cancers encompass a very large group of tumours, we used childhood cancer histopathology images to develop and test our system. Our computational experiments demonstrate that the synthetic images significantly enhances performance of various AI classifiers. 

## Example Results



### Real and Synthetic Images
<img src="imgs/synthetic_main_figure.jpg" width="800px"/>


### Real WSIs and CAM Result
<img src="imgs/GradCamFigure.png" width="1200px"/>

## Dataset

In this study, we conducted experiments using histopathological whole slide images(WSIs) of five rare childhood cancer types and their sub-types, namely ependymoma (anaplastic, myxopapillary, subependymoma and no-subtype), medulloblastoma (anaplastic, desmoplastic and no-subtype), Wilms tumour, also known as nephroblastoma (epithelial, blastomatous, stromal, Wilms epithelial-stromal, epithelial-blastomatous and blastomatous-stromal), pilocytic astrocytoma and Ewing sarcoma.

Tumour histopathology WSIs are collected at [Ege University](https://med.ege.edu.tr/eng-2025/education.html), Turkey and Aperio AT2 scanner digitised the WSIs at 20× magnification. WSIs will be available publicly soon

## Prerequisites
- Linux (Tested on Red Hat Enterprise Linux 8.5)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 Ti x 4 on local workstations, and Nvidia A100 GPUs on [TRUBA](https://www.truba.gov.tr/index.php/en/main-page/)
- Python (3.9.7), matplotlib (3.4.3), numpy (1.21.2), opencv (4.5.3), openslide-python (1.1.1), openslides (3.4.1), pandas (1.3.3), pillow (8.3.2), PyTorch (1.9.0), scikit-learn (1.0), scipy (1.7.1),  tensorboardx (2.4), torchvision (0.10.1).

### Getting started

- Clone this repo:
```bash
git clone https://github.com/ekurtulus/SCI-AIDE.git
cd SCI-AIDE
```

- Install PyTorch 3.9 and other dependencies (e.g., PyTorch).

- For pip users, please type the command `pip install -r requirements.txt`.

- For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

### Synthetic Images Generation

<img src="imgs/gif_1.gif" width="1200px"/>

- Clone FastGAN repo:
```bash
git clone https://github.com/odegeasslbc/FastGAN-pytorch.git
cd FastGAN-pytorch
```
- Train the FastGAN model:
```bash
python classifer.py --path $REAL_IMAGE_DIR --iter 100000 --batch_size 16
```
- Inference the FastGAN model:
```bash
python eval.py --ckpt $CKPT_PATH --n_sample $NUMBERS_OF_SAMPLE
```

- Train the SCI-AIDE model:
```bash
python train.py --datapath $DATAPATH_PATH --model $MODEL --savepath $SAVING_PATH --task $TRAINING_TASK
```

The list of other arguments is as follows:

- --lr : Learning rate (default: 5e-5)
- --opt : Optimizers ( "Adam", "SGD", "RMSprop", "AdamW" , default= "SGD")
- --batch-size : Batch size (default: 32)
- --halftensor : Mixed presicion acivaiton
- --epochs : Numbers of epochs
- --scheduler : Learning scheduler ( "cosine", "multiplicative" , default="cosine")
- --augmentation : Augmentation selection ( "randaugment", "autoaugment", "augmix", "none", default= "randaugment" )
- --memory : Data reading selection ( "none", "cached", default= "none" )


- Evaluation the SCI-AIDE model:
```bash
python wsi_attention.py --datapath $DATAPATH_PATH --model $MODEL --model_weights $MODEL_WEIGHT --output $OUTPUT_PATH --name $NAME --num_classes $NUM_CLASSES
```
The list of other arguments is as follows:

- --attention_level : ("pixel", "patch", default="patch)
- --cam : CAM selection ( "GradCAM", "ScoreCAM", "GradCAMPlusPlus", "AblationCAM", "XGradCAM", "EigenCAM", "FullGrad", default="EigenCAM" )

- Diagnosis WSI with the SCI-AIDE model:
```bash
python wsi_diagnosis.py --task $DIAGNOSIS_TASK --datapath $WSI_PATH --output $OUTPUT_PATH --config $CONFIG_FILE_PATH --name $NAME
```
The list of other arguments is as follows:

- --overlap : Patches overlaping raito (default :0 )
- --patch_size : WSI oatching size (default : 1024 )
- --heatmap : Heatmap inference activation
- --white_threshold : White pathch elimiantion ration (default :0.3)

### Apply a pre-trained SCI-AIDE model and evaluate
For reproducability, you can download the pretrained models for each algorithm [here.]()

## Issues

- Please report all issues on the public forum.

## License

© [DeepMIA Lab](https://deepmia.boun.edu.tr/) This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Funding

This work was funded by [TUBITAK](https://www.tubitak.gov.tr/) for International Fellowship for Outstanding Researchers.


## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```

```

### Acknowledgments
Our code is developed based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/tree/54a6cca27a9a3e092a07457f5d56709da56e3cf5). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation, and [FastGAN-pytorch](https://github.com/odegeasslbc/FastGAN-pytorch) for the PyTorch implementation of FastGAN used in our single-image translation setting.
