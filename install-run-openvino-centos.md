# Installing OpenVINO

1. Install Intel OpenVINO toolkit - https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux 
2. After installing run:
```
  source openvino_install_location/bin/setupvars.sh
  yum install gtk3
```

3. Build samples:
```
  cd cvsdk_install_location/deployment_tools/inference_engine/samples
  mkdir build
  cd build
  cmake ..
  make
  
  source <cvsdk_intall_location>/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_caffe.sh
  source <cvsdk_intall_location>/deployment_tools/model_optimizer/install_prerequisites/../venv/bin/activate
  ```
  
4. Download and unzip benchmarking tool - http://nnt-srv01.inn.intel.com/Users/drizshko/samples/bench.zip
5. Build benchmarking tool
```
  mkdir build
  cd build
  cmake ..
  make
```
If an error comes up, add the following lines to dldt.cmake and repeat Step 5.
```
vi <bench_location>/cmake/dldt.cmake
set(ENV{INTEL_CVSDK_DIR} "/home/agartia/intel/computer_vision_sdk_2018.1.249")
set(ENV{IE_PLUGINS_PATH} "/home/agartia/intel/computer_vision_sdk_2018.1.249/deployment_tools/inference_engine/lib/centos_7.3/intel64")
```
6. By default cmake file have ENABLE_OPENCV = ON and ENABLE_OPENCV_VIDEO = OFF.\
ENABLE_OPENCV - lets to use images as input.\
With ENABLE_OPENCV = OFF only synthetic input will be available.\
ENABLE_OPENCV_VIDEO - lets to use video files as input, will work only with ENABLE_OPENCV enabled.

To disable OpenCV support use
```
  cmake -DENABLE_OPENCV=OFF ..
```
7. Run application with -h flag
```
  Options:

    -h                           Print a usage message.
    -i <path>                    Path to input video file or directory with input images. Without this parameter synthetic images will be used.
    -prefetch                    Prefetch number of images. Optional.
    -m <path>                    Path to IR .xml file. Required.
    -w <path>                    Path to IR .bin file. Optional if bin file has same name as model file.
    -input <path>                Name of input layer, required for models with more than 1 inputs.
    -d CPU|GPU|HETERO:FPGA,CPU   Device name. CPU by default.
    -l <path>                    Path to plugin extension. Optional.
    -ic                          Number of images to infer. Optional.
    -b                           Batch size. 1 by default. Optional.
    -ni                          Number of iterations. 1 by default. Optional.
    -warm                        Number of warm-up iterations. 0 by default. Optional.
    -pc                          Print performance counters. Optional.
    -summary                     Print model stats. Optional.
    -memory                      Print memory stats. Optional.
```
8. Imagenet unzip:
```
for i in *.tar; do mkdir "${i%.*}"; tar -xf $i -C "${i%.*}"; done
```
9.   MPI mode:
```
mpirun -np 4 benchmark -m path_to_model -ic number_of_syncth_images -mpi
```
10. Runner Script:
```
python runner.py –ap ./benchmark –m model_path –i images_path -ic 1024 –c 48 –b 1,2,3,4,8,16 –si 1 –mpi 1 –ht 1 -o out.csv

  “ap” – path to benchmark application
  “i” – path to image folder, if not set synthetic images will be used
  “ic” – number of images for inference
  “c” – number of cores to use, will be used to find optimal number of processes. Eg: -c 48 will try all possible multipliers in MPI mode: 1,2,3,4,6,8,12,16,24,48
  “b” – list of batches. Eg: -b 1,2,3,4,8,16
  “si” – run on single instance mode
  “mpi” – run on MPI mode
  “ht” – if enabled will set KMP_AFFINITY=granularity=fine,compact,1,0, or if no HT set KMP_AFFINITY=
  “o” – path to csv file to save results

 ``` 
 FYI:
 -- Supported Layers in Model Optimizer: https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer#intermediate-representation-notation-catalog
 -- Create Custom layers in Tensorflow for MO: 
 https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer#inpage-nav-4-2
 https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer#Custom-Layers
 
 # Example Commands to run OpenVINO
 
 ## Model Optimizer
 ```
 ./ModelOptimizer -p FP32 -w ../intel_models/resnet50/caffe/resnet_50_16_nodes_2k_batch.caffemodel -d ../intel_models/resnet50/caffe/resnet_50_16_nodes_2k_batch.prototxt -i -b 1

python mo_tf.py --input_model  ~/DLDT/model_ir/MobilenetV2/mobilenet_v2_1.4_224_frozen.pb --input_shape '(1,368,368,3)'

python mo_tf.py --input_model ~/DLDT/model_ir/openpose/graph_freeze.pb --input_shape [1,368,368,3] 

python mo_tf.py --input_model ~/DLDT/model_ir/openpose/openpose_mobilenet.pb --input_shape [1,368,368,3],[1,368,368,3],[1,368,368,3],[1,368,368,3],[1,368,368,3],[1,368,368,3] --input 'Placeholder','Placeholder_1','Placeholder_2','Placeholder_3','Placeholder_4','Placeholder_5'

python mo.py \
--framework caffe \
--output_dir model_ir \
--batch 1 \
--data_type FP32 \
--log_level INFO \
--input_model ~/intelcaffe_skl/models/resnet50_cvgj/resnet50_cvgj_iter_1000.caffemodel \
--input_proto ../intel_models/resnet50_cvgj/resnet50_cvgj.prototxt

```

## Inference Engine
```
./benchmark \
-d CPU -ni 100 -pc -summary -b 1 \
-i ~/caffe_data/examples/imagenet/ilsvrc12_val_lmdb/data.mdb \
-m ~/intel/computer_vision_sdk_2018.0.234/deployment_tools/model_optimizer/model_ir/ResNet-50_fp32.xml

./benchmark -i ~/caffe_data/ILSVRC2012_img_train_t3/n02110063 -d CPU -ni 100 -pc -summary -b 1 -m ~/intel/computer_vision_sdk_2018.0.234/deployment_tools/model_optimizer/model_ir/ResNet-50_fp32.xml

./benchmark -m ~/intel/computer_vision_sdk_2018.0.234/deployment_tools/model_optimizer/model_ir/resnet50/ResNet-50_fp32-INT8.xml -d CPU -ic 1024 -pc -summary -b 1 | tee resnet_syn_b1.log

./benchmark -m ~/intel/computer_vision_sdk_2018.0.234/deployment_tools/model_optimizer/model_ir/vgg/vgg16.xml -d CPU -ni 10 -pc -summary -b 1 -ic 10| tee vgg_b1.log
```

## Log generation
```
for f in ls *.log; do cat $f | grep "Loading" | tee -a combined.log; cat $f | grep "Batch size" | tee -a combined.log; cat $f | grep "Images per second" | tee -a combined.log; done
```

## Runner Script
```
python runner.py -ap /home/agartia/DLDT/bench_june18/bin/intel64/Release/benchmark -m /home/agartia/DLDT/model_ir/MobilenetV2/mobilenet_v2_1.4_224_frozen.xml -ic 1024 -c 36 -b 1,2,4,8,16,32,64 -si 0 -mpi 1 -ht 1 -o logs/skl_6140_DLDT_inference_Mobilenet_multisocket.csv

python runner.py -ap /home/agartia/DLDT/bench_june18/bin/intel64/Release/benchmark -m /home/agartia/DLDT/model_ir/resnet50/ResNet-50_fp32-INT8.xml -ic 1024 -c 36 -b 1,2,4,8,16,32,64 -si 0 -mpi 1 -ht 1 -o logs/skl_6140_DLDT_inference_resnet50_multisocket.csv

python runner.py -ap /home/agartia/DLDT/bench_june18/bin/intel64/Release/benchmark -m /home/agartia/DLDT/model_ir/inceptionV3/Inception_V3.xml -ic 1024 -c 36 -b 1,2,4,8,16,32,64 -si 0 -mpi 1 -ht 1 -o logs/skl_6140_DLDT_inference_inceptionV3_multisocket.csv

python runner.py -ap /home/agartia/DLDT/bench_june18/bin/intel64/Release/benchmark -m /home/agartia/DLDT/model_ir/vgg/vgg16.xml -ic 1024 -c 36 -b 1,2,4,8,16,32,64 -mpi 1 -ht 1 -o logs/skl_6140_DLDT_inference_vgg16_multisocket.csv
```
 

 
 
