Based on the paper [Learning in the Frequency Domain](https://arxiv.org/abs/2002.12416)

Packages:
- [libjpeg](https://libjpeg-turbo.org/) 
- [jpeg2dct](https://github.com/uber-research/jpeg2dct)
- [Hydra](https://github.com/facebookresearch/hydra)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

Datasets:
- [TinyImagenet](https://www.kaggle.com/c/tiny-imagenet/data)
- [Imagenette](https://github.com/fastai/imagenette)

File Overview:
- conf
    - conf.yaml: stores hyperparameters and variables for training. Any value specified in conf.yaml can be replaced at runtime by
    specifying the new value as follows
    
        `python train.py {parameter}={new value}`
        where parameter is the parameter you want to replace and new value is the new value
 - datasets
    - calculate_mean_std.py: Computes the mean and standard deviation in the frequency domain
    - create_dataset.py: Creates datasets based on PyTorch's dataset format. Dataset classes include 
        - TinyImagenetTrain
        - TinyImagnetTest
        - TinyImagenetTrainRGB
        - TinyImagenetTestRGB
        - ImagenetteTrain
        - ImagenetteTest
    - make_train_val_split.py: Splits training set into train and validation set and stores the values in JSON files
    - transforms.py: Contains transforms that operate in the frequency domain
 - models
    - resnet.py: Contains default source code for ResNet from PyTorch and modifications that allow data in the frequency domain to be used
 - avgs.npy: numpy array that contains (y channel average,cb channel average, cr channel average) after the discrete cosine transform for Tiny-Imagenet
 - avgs_imagenette.npy: numpy array that contains (y channel average,cb channel average, cr channel average) after the discrete cosine transform for Imagenette 160
 - avgs_imagenette_320: numpy array that contains (y channel average,cb channel average, cr channel average) after the discrete cosine transform for Imagenette 320
 - stds.npy: numpy array that contains (y channel average standard deviation, cb channel average standard deviation, cr channel average standard deviation) after the discrete cosine transform for Tiny-Imagenet
 - stds_imagenette.npy: numpy array that contains (y channel average standard deviation, cb channel average standard deviation, cr channel average standard deviation) after the discrete cosine transform for Imagenette-160
 - stds_imagenette_320.npy: stds.npy: numpy array that contains (y channel average standard deviation, cb channel average standard deviation, cr channel average standard deviation) after the discrete cosine transform for Imagenette-320
 - frequencies.npy: array of number of times the different channels were activated in Imagenette-160
 - train.py: main training file for images in the frequency domain. To run:
    `python train.py`
 - train_list.json: Dictionary that contains the classes and the images from those classes that are part of the training set for Tiny-Imagenet
 - train_list_imagenette.json: Dictionary that contains the classes and images from those classes that are part of the training set for Imagnette (160 and 320)
 - val_list.json: Dictionary that contains the classes and the images from those classes that are part of the validation set for Tiny-Imagenet
 - val_list_imagenette.json: Dictionary that contains the classes and the images from those classes that are part of the validation set ofr Imagenette(160 and 320)
 - train_rgb.py: main training file for images in the RGB domain. To run
    `python train_rgb.py`
  
 
        
         

    
 Sources:
 - Used as reference: https://github.com/calmevtime/DCTNet 
    - datasets/calculate_mean_std.py followed the method done at the bottom of done at the bottom of https://github.com/calmevtime/DCTNet/blob/master/classification/datasets/dataset_imagenet_dct.py
    - models/resnet.py
        - lines 323-326 were taken from lines 397-407 from here: https://github.com/calmevtime/DCTNet/blob/master/classification/models/imagenet/resnet.py
        - lines 10-286 are the source code for PyTorch ResNet
 - Used for center crop: https://medium.com/curious-manava/center-crop-and-scaling-in-opencv-using-python-279c1bb77c74 
  