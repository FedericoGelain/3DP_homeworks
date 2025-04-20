Structure from motion
========================================

---
## Instructions
**Prerequisites (in debian-based distro)**

    sudo apt install build-essential cmake libboost-filesystem-dev libopencv-dev libomp-dev
    sudo apt install libceres-dev libyaml-cpp-dev libgtest-dev libeigen3-dev

**Building (Out-of-Tree)**

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    
**Usage (from bin/ directory)**

    ./matcher <calibration parameters filename> <images folder filename> <output data file> [focal length scale]
    ./basic_sfm <input data file> <output ply file>

**Datasets**
The /dataset folder contains the two provided datasets and the one acquired using the phone camera, the corresponding
camera parameters files and the resulting point clouds reconstructed for all of them

**Examples**

    ./matcher ../datasets/3dp_cam.yml ../datasets/images_1 data1.txt 1.1
    ./matcher ../datasets/3dp_cam.yml ../datasets/images_2 data2.txt 1.1
    ./matcher ../datasets/cam_phone.yml ../datasets/ducks data_ducks.txt 1.1

    ./basic_sfm data1.txt cloud1.ply
    ./basic_sfm data2.txt cloud2.ply
    ./basic_sfm data_ducks.txt cloud_ducks.ply
---