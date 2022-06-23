# Deep Hashing

The simple approach of retrieving a closest match of a query image
from one in the gallery, compares an image pair using sum of
absolute difference in pixel or feature space. The process is computationally
expensive, ill-posed to illumination, background composition,
pose variation, as well as inefficient to be deployed on gallery
sets with more than 1000 elements. Hashing is a faster alternative
which involves representing images in reduced dimensional simple
feature spaces. Encoding images into binary hash codes enables
similarity comparison in an image-pair using the Hamming distance
measure. The challenge however lies in encoding the images
using a semantic hashing scheme that lets subjective neighbours
lie within the tolerable Hamming radius. This work presents a solution
employing adversarial learning of a deep neural semantic
hashing network for fashion inventory retrieval. It consists of a
feature extracting convolutional neural network (CNN) learned to
(i) minimize error in classifying type of clothing, (ii) minimize hamming
distance between semantic neighbours and maximize distance
between semantically dissimilar images, (iii) maximally scramble
a discriminator’s ability to identify the corresponding hash codeimage
pair when processing a semantically similar query-gallery
image pair. Experimental validation for fashion inventory search
yields a mean average precision (mAP) of 90.65% in finding the
closest match as compared to 53.26% obtained by the prior art of
deep Cauchy hashing for hamming space retrieval. [Singh S. et al]

## Network Architecture

![Full architecture of the network used](media/fullarchitecture.jpg)
this architecture is divided into 3 components:
1) The Encoder
2) The Classifier
3) The Discriminator

## Results
## Model
## Demo
## Setup
## Train
## Code and Directory Organisation
## Code Structure
## Creating Dataset Directory Tree
## How to Perform Prediction
## Acknowledgement
## References


Singh, S., Sheet, D., & Dasgupta, M. (2019). Adversarially trained deep neural semantic hashing scheme for subjective search in fashion inventory. arXiv preprint arXiv:1907.00382.