# ABFS - Automatic Building Footprint Segmentation

![Example Truth Mask](https://user-images.githubusercontent.com/634167/51509050-d45ae400-1dc5-11e9-9dbc-04c64eed4e96.jpg)

### Outline

* [Project Description](#project-description)

## Project Description

As satellite imagery has improved over the past ten years, applications of aerial images have dramatically increased. From crop yield to war devastation, these images can tell a detailed story even without consideration across a temporal spectrum. One type of data implicitly present in most urban and suburban areas is a building footprint.

The area that a building makes up can give rough estimates for both specific and general applications. General analysis of building footprints can give a sense of house size and enable broad cross-section comparison of regions. More specifically and centrally to this project, determining roof area from building footprints can give a rough estimate of potential work. Although a user can outline these manually, one-click or automatic building area would give insight even faster and further incentivize users to proceed with a purchase. One such example of building footprints is Google’s Project Sunroof that only requires an address. From there, it draws the building footprint and performs the relevant analysis to produce an estimate of how much solar power could be obtained and how much money might be saved.

My personal motivation for this project stems from a client application that currently requires manual outlining the footprint in order to produce a roof estimate. Although this project’s data will be entirely open source based and I will own the source code, its final performance may be tested using that client’s customer footprint outlines as per an agreement with the company I work for—RoleModel Software.

### Primary Task

To solve this problem, I am proposing a project that encompasses the entire data science workflow (if possible) from exploratory data analysis to model deployment. It will employ both regression and classification machine learning techniques as they overlap with a supervised segmentation approach that identifies building footprints.

### Data

Assuming the data is in keeping with its description, I plan to employ the building footprint labels available from SpaceNet. This data, available through AWS, was initially publicized last year during a Kaggle competition and has been entirely released to the public on a requester-pays basis.

The SpaceNet data includes six areas of interest with each dataset averaging over 100,000 building labels. The largest one, Rio, includes a region of over 2,500 square kilometers with 382,000 polygons in GeoJSON as well as the associated tiles. Since the dataset was specifically designed for machine learning research, they document several available Python scripts for common data transformations.

### Analysis Plan

Having done a little bit of computer vision analysis using Deep Learning last year, I plan to use a segmentation neural network at least for the initial modeling process. Because U-Nets seem to have decent success as fully-convolutional solutions to related problems, I’ll use that as my starting point for analysis. If time allows, I may investigate generative adversarial networks or generative ladder networks as alternate model topologies.

Using the typical approach, I’ll split the data into train, test, and validation sets with the additional potential of doing final performance evaluation using the shapes manually annotated in RoleModel Software’s relevant client project. As far as general tools, TensorBoard will likely be the central feedback system for accuracy in both the train and validation sets. To minimize potential change from my work last year that might distract from the primary modeling solution, I’ll begin with Keras and Tensorflow running on my custom build with two 1080 TI’s but will stay open-minded if other frameworks prove more appropriate.

### Anticipated Risks (Examples)

* _Initial neural network topology is difficult to integrate, doesn’t work well, or otherwise seems broken._ Perform initial research to not become too locked down to a model too early.
* _Performance on test data doesn’t match accuracy metrics during training._ Using TensorBoard’s image callback should give a visual indication of how the segmentation progresses.
* _Data sets (test/train/val) bleed into one another, or accuracy metrics don’t seem consistent with example results._ By writing unit tests where possible, these subtle bugs shouldn’t surface.
* _Clipped satellite imagery may throw off accuracy._ For at least the initial passes, filtering out these examples may be perfectly acceptable.
* _Model does not generalize well to client data._ Properly running validation metrics and then testing with several actual use case scenarios should mitigate this issue.
* _Other issues…_ Regularly review/revise timeline as needed to always keep the project’s scope in perspective.

### Timeline

* Week 1: Exploratory Data Analysis of SpaceNet
* Week 2: Setup Keras/Tensorflow project infrastructure: Tensorboard, generator, etc.
* Week 3: Initial modeling comparison of FCNs (2+): U-Net, FCN, DeepLab, SegNet
* Week 4: Iterate model hyper-parameters using most promising network topology
* Week 5: Develop Python API for using model in production (continue model tweaking)
* Week 6: Model alternates (Mask-RCNN, pix2pixHD?) OR iterate model params
* Week 7: Deploy API and conduct model performance testing (accuracy metrics)
* Week 8: Final tweaks, blog post on rcd.ai, video presentation

### Helpful Links

* http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
* https://sthalles.github.io/deep_segmentation_network 
* https://github.com/mrgloom/awesome-semantic-segmentation
