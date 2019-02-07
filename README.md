autoencoder-showcase
====================

**Welcome to the supplementary experiments for the paper 'A Showcase
of the Use of Autoencoders in Feature Learning Applications'**

Install this package using:

```r
devtools::install_github("ari-dasci/autoencoder-showcase")
```

Inside this package, you will find 4 main functions:
  - `anomaly_detection()`  
    Creates a synthetic multi-valued time series with an anomalous
    region and performs anomaly detection.
  - `visualization()`  
    Downloads the Statlog dataset and compacts it to 2 and 3
    dimensions for visualization.
  - `hashing()`  
    Loads IMDB dataset from Keras, trains an autoencoder and
    hashes the test subset, measuring the correspondance between
    distance among instances and Hamming distance among their
    hashes.
  - `denoising()`  
    Loads CIFAR10 dataset and trains a denoising autoencoder,
    performs denoising over the test subset.
