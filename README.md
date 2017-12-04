# VOCUS2 computational visual saliency system
This program computes a saliency map for a given input image or video.

Input|Output
------------ | -------------
![Input](/images/test-image-MSRA-3_97_97769.jpg?raw=true) | ![Saliency map](/images/test-image-MSRA-3_97_97769_saliency.jpg?raw=true)

A detailed description may be found in the following publication:
"Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

If you find this software useful in your research, please cite the paper.

BiBTeX:
```
@InProceedings{Frintrop_2015_CVPR,
	author = {Frintrop, Simone and Werner, Thomas and Martin Garcia, German},
	title = {Traditional Saliency Reloaded: A Good Old Model in New Shape},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2015}
} 
```

This software was implemented by Thomas Werner, designed and supervised by Simone Frintrop.

## 1. Installation requirements
* gcc 4.7 
* OpenCV 2.4.7 or higher
* Boost C++ libraries 1.46.1

This code was tested on Linux Machines only.


## 2. Compiling
Compile the program with:
```
cmake . 
make
```
## 3. Execution
A simple call for a test image looks like:
```
./vocus2 test_img.png
```
The output is written as a PNG to a subfolder named `saliency/` in the location of the input file. 
Saving the output for video files is not supported yet.

Calling the program this way uses default parameters that are
reasonable especially for images with large objects, such as the ones from the MSRA salient object database (see Sec. 6). For real-world applications with cluttered scenes and many small objects, e.g. in robotics, use instead the parameters from the `coffee_cfg.xml` config file (see Sec. 6).

A call of `./vocus2 -h` will show a list of all tunable parameters along with their default values and a short description. If any parameters are given, they must appear before the input files:
```
./vocus2 [parameters] test_img.png
```
To process a set of images, a series of file names can be given to the program.
To use all images in directory `/img`:
```
./vocus2 [parameters] img/*.png
```
Or specify files separately:
```
./vocus2 [parameters] img/img1.png img/img2.png ...
```
Without a specified output path, the corresponding saliency maps are written to `img/saliency/<imageName>_saliency.png`.


## 4. Parameters
* `-x <name>`	Loads a config given by `<name>`. For more information on Config Files, see Section 6.
* `-C <value>`	Specifies which colorspace should be used for the computation. Possible values:

   				0: Lab
				1: Opponent Colorspace (CoDi), like in Klein/Frintrop DAGM 2012
				2: Opponent Colorspace (Equal domains), like in Klein2012 but shifted and scaled to [0:1]

				Default: 1

* `-f <value>`	Specifies how feature maps should be fused to generate the conspicuity maps. Possible values:

				0: Arithmetic mean (well suited for salient object segmentation).
				1: Max
				2: Uniqueness weight. Each maps is weighted according to its uniqueness before 
				   fusing takes place. This means a feature maps with many salient peaks will get a low 
				   weight whereas a map with a single (unique) salient peak will get a high weight  
				   (well suited for determining the order in which regions should be investigated 
				   and for prediction eye movements. (details see: Frintrop, PhD thesis, 2005)

				Default: 0
			
* `-F <value>`	Same as `-f`. Specifies how conspicuity maps are fused. Default: 0

* `-p <value>`	Defines the pyramidal structure that is used for the center-surround computation. Since all three options
			produce similar results it is best to use the default value. Possible values:

				0: Two independent pyramids (Classic)
				1: Two pyramids derived from a single base pyramid (CoDi-like)
				2: Surround pyramid derived from the center pyramid (Frintrop et al, CVPR 2015)

				Default: 2

* `-l <value>`	Defines the first pyramid layer that is used. A value of 0 means that the start layer is the same size as the input image and will produce a detailed saliency map. With increasing value the level of detail will decrease. Values have to be positive integers or 0 and must not be larger than the stop layer defined by `-L`. 

			Default: 0
						
* `-L <value>`	Defines the last pyramid layer that is used. Important: `L >= l`. The further `l` and `L` are
   			apart, the more layers are used and the higher the computation time. Default: 4

* `-S <value>`	Defines the number of scales for each pyramid layer. Increasing `S` can increase the
			level of detail but also the computation time. Values must be positive integers.

			Default: 2
	
* `-c <value>`	Defines the sigma of the Gaussian that is used to produce the center pyramid. The value could be either an integer or float and must be positive. The value is directly related to the object size. A small value should be used to detect small objects and a large value for large objects. 

			Default: 3.0

* `-s <value>`	Specifies the sigma of the Gaussian for the surround pyramid. Values can be float or integer but have to be > 0 and > center sigma. 

			Default: 13


* `-e`		Use Combined Feature [default: off]
		Default (off) treats all channels of the color space equally.
		Switching `-e` on results in fusing the color channels RG and BY first, and fusing then the
		resulting map with the intensity map (as done in Itti et al. PAMI 1998).
		This point corresponds to version 3 in sec. 4.1 of our CVPR 2015 paper.

* `-v <id>`		Activates a webcam with the given ID as input. If only a single webcam is connected the ID should be 0.

* `-V`		Indicates that video files are used instead of images.

* `-t <value>`	Specifies the region growing threshold for the selection of the fixation. This has
			no effect on the saliency computation and is only used for visualization. The
			threshold determines the size of the most salient blob (the smaller, the larger).

			Default: 0.75

* `-N`		Turns off visualization. Should be used when batch processing multiple files so that
			the process isn't blocked.

* `-o <path>`	Write results to specified path 
			Default: `<input_path>/saliency/*`

* `-w <path>`	Write all intermediate maps to an existing directory. The directory has to exists beforehand.

* `-b` 		Add center bias to the saliency map




## 5. Example calls
To detect quite large objects (e.g. in the MSRA Dataset), use layer 0-4 with 2 scales per layer (default), a center 
size of 3 and a surround of size 13:
```
./vocus2 -c 3 -s 13 image.png
```
If the objects are smaller, reduce center size:
```
./vocus2 -c 2 -s 13 image.png
```
If this gives to many details, use layers 1-5 instead:
```
./vocus2 -l 1 -L 5 -c 2 -s 13 image.png
```
If the objects are close to each other, it might be best to reduce the surround size.
```
./vocus2 -l 1 -L 5 -c 2 -s 5 image.png
```
Use the LAB colorspace instead of the default opponent space:
```
./vocus2 -C 0 image.png
./vocus2 -C 0 -l 1 -L 5 -c 2 -s 5 image.png
```

## 6. Usage of Config Files
The config files, in the `cfg/` directory, contain parameter sets that are optimized for different datasets.

When a config file is given, any additional parameter will have a higher priority and override the corresponding parameter from the file. 

We provide the following sample configurations:
- `msra_cfg`: for the MSRA Salient Object Database, this parameter set works also well for other benchmarks for salient object segmentation, e.g., the SED1 and SED2, the ECSSD dataset and the PASCAL-S dataset.

* `coffee_cfg`: for the [Coffee Machine Sequence](http://www.iai.uni-bonn.de/~martin/datasets.html) and 
* `kitchen_cfg`: for the [Kitchen Object Discovery Dataset](http://www.mmp.rwth-aachen.de/projects/kod/).

The last two config-files are useful for data in real-world applications with cluttered scenes containing many, small objects, e.g. in robotics.