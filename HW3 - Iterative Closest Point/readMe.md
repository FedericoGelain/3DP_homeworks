Iterative Closest Point
========================================

---
## Instructions

**Building (Out-of-Tree)**

    mkdir build
    cd build
    cmake ..
    make
    
**Usage (from build/ directory)**

    ./registration <source> <target> mode
where mode is either "svd" or "lm"

**Datasets**

In the */data* folder, each one of the three directories *bunny*, *dragon* and *vase*
contains the source and target point clouds to perform ICP registration.

**Examples**

    ./registration ../data/bunny/source.ply ../data/bunny/target.ply svd
    ./registration ../data/dragon/source.ply ../data/dragon/target.ply svd
    ./registration ../data/vase/source.ply ../data/vase/target.ply svd
    
    ./registration ../data/bunny/source.ply ../data/bunny/target.ply lm
    ./registration ../data/dragon/source.ply ../data/dragon/target.ply lm
    ./registration ../data/vase/source.ply ../data/vase/target.ply lm

**Results**

The */data/registration_results* folder contains the resulting point cloud 
obtained after performing the ICP registration with either SVD and LM for all
the three datasets.

---
