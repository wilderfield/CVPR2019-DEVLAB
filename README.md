# Xilinx ML Suite on AWS @ CVPR : Developer Lab

Today we will walk through how to use Xilinx's [ML Suite 1.4](https://github.com/Xilinx/ml-suite) to target Xilinx FPGAs for deployment.  

We have the following sections:  

1. [Jupyter Notebook: ML Suite Caffe Flow w/ Pretrained Model](#section-1-ml-suite-caffe-flow-w-pretrained-model)
2. [CLI: Benchmark The Performance of a Model](#section-2-benchmark-the-performance-of-a-model)
3. [Jupyter Notebook: ML Suite Caffe Flow End2End w/ MNIST Dataset](#section-3-ml-suite-caffe-flow-end2end-w-mnist-dataset)
4. [CLI: Real Application - Face Detection in Xilinx FPGAs on AWS](#section-4-real-application---face-detection-in-xilinx-fpgas)
  
All sections require you to be running the Xilinx ML Suite Caffe Docker container provided in this instance.
  
Should you need to refer back to this instruction while in the container, you can view it in /opt/ml-suite/share/README.md  

# Starting the Container
Xilinx has provided a bash script to start the container, and mount the necessary system resources.  
It also creates a directory `/home/ubuntu/share` which can be used to easily pass files back and forth between the host OS, and the running docker container.  
Inside the container, the directory is mounted as `/opt/ml-suite/share`  
By default the container will be removed upon exit, so any changes you make outside of the `/opt/ml-suite/share` directory will be lost.  
To change this behavior, you can remove the --rm flag in `docker_run.sh`  
  
```
$ pwd
/home/ubuntu
$ ./docker_run.sh
```

# Section 1: ML Suite Caffe Flow w/ Pretrained Model
For this part of the lab, you will use a Jupyter Notebook.  
This requires launching a Jupyter Notebook server from within the docker container.  
This notebook requires that you download some pretrained models provided by Xilinx, and will require a minimal version of the IMAGENET 2012 Dataset.  
The IMAGNET images will be used for quantization, as well as for testing the quantized model.  
  
1. Start the container in interactive mode, and enter it.  
  `$ ./docker_run.sh `  
2. Download the first 500 images of the IMAGENET 2012 validation set, and their corresponding labels.  
  ```
  python -m ck pull repo:ck-env  
  python -m ck install package:imagenet-2012-val-min
  python -m ck install package:imagenet-2012-aux    
  head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
  ```
3. Resize the downloaded pictures to a common dimension (Script does 256x256).  
  `$ cd /opt/ml-suite/examples/caffe/ && python resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256 `  
4. Download a few common models.  
  `$ cd /opt/ml-suite/examples/caffe/ && python getModels.py `  
5. Start Jupyter Notebook Server  
  `$ cd /opt/ml-suite/notebooks && jupyter notebook --no-browser --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' `  
6. Navigate in a web browser to the Jupyter Notebook Server.  
  ```
  # You should use the Public DNS(IPv4) of your instance and the default port of 8888 
  # i.e. http://ec2-3-94-181-134.compute-1.amazonaws.com:8888
  ```
7. Step through the notebook `image_classification_caffe.ipynb`  
  ```
  The notebook will walk through the steps required to deploy a Caffe Model on the FPGA.  
  You can use the drop down cell to select different models.  
  We are able to directly run on the FPGA using Caffe's python bindings, as they provide support for custom Python layers.  
  We replace a portion of the original model (a.k.a.a subgraph) with the custom Python layer which will exercise the FPGA.  
  Layers not supported by the FPGA are directly ran by Caffe on the CPU.  
  Using this method, we entirely abstract away the ML Suite Python APIs. 
  While this is a beautiful solution, it may not always be the best method for maximum performance.
  Eventually, you may need to directly use the Python APIs exposed in /opt/ml-suite/xfdnn/rt/xdnn.py  
  ```
8. Stop your Jupyter Notebook Server (Don't exit the container!!!)
  `CTRL+C twice, should kill the server`


# Section 2: Benchmark The Performance of a Model  
For this part of the lab, you will use some provided scripts to check the performance of the model you prepared in Section 1.  
You should have a quantized & compiled model in `/opt/ml-suite/notebooks/work/`  
The artifacts will correspond to whichever model you chose in the notebook's drop down cell.  
In this step we will simply feed these artifacts into a script you can use for benchmarking.  
Note that these scripts only work for models that end in a single FC, and softmax, but can be used as starting points for other networks.  
  
To unpack the switches we are using here...  
```  
-cn provides a custom compiler json to mp_classify.py : This is the schedule of operations to be performed on the hardware.
-cq provides a custom quantizer json to mp_classify.py : This contains scalars for the affine mapping between FP32 and INT8.
-cw provides a custom weights blob to mp_classify.py : These are really just the original FP32 weights packaged up nicely in HDF5.
-d provides a directory of images to iterate over
-t provides the script we will run, streaming_classify_fpgaonly will select mp_classify, and set up some arguments for benchmarking
-x tells mp_classify to run forever
-v tells mp_classify to stream out debug prints
-ns tells mp_classify to use a certain number of streams. This boils down to, how many images will you enqueue in the host before sending them to hardware.  
     Enqueing 2 images per PE is a good way to make sure that the hardware is always busy and throughput is maximized, although this will increase system latency.  
```

1. Feed your prepared model into `mp_classify.py`  
  ```
  $ cd $MLSUITE_ROOT/examples/deployment_modes && \
    ./run.sh -cn ../../notebooks/work/compiler.json \
             -cq ../../notebooks/work/quantizer.json \
             -cw ../../notebooks/work/deploy.caffemodel_data.h5 \
             -d $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min \
             -t streaming_classify_fpgaonly -x -v -ns 2 | python $MLSUITE_ROOT/xfdnn/rt/scripts/speedometer.py  
  ```
2. View the report.
  ```
  # Should See Something Like This (ResNet50) w/ 2 streams
   
  --------------------
  XDNN pipeline report
  --------------------
  
   quant+format                     |   0.38 (min: 0.34)
         ddr_wr                     |   0.30 (min: 0.01)
         submit                         0.03 (min: 0.02)
         fpga_0                         0.00 (min: 0.00)
         fpga_1  ||||||||||||||||||||   3.82 (min: 3.80) x
         ddr_rd                         0.09 (min: 0.06)
           post                         0.03 (min: 0.02)
  
  Input rate          : 267 images/s
  Max FPGA throughput : 261 images/s with 1 PEs (pre-/post-processing not included)
  FPGA utilization    : 100.00%
  End-to-end latency  : 7.16 ms (FPGA is 66% oversubscribed)
  ``` 
3. Kill the benchmark scripts with: CTRL+Z; kill -9 %%

# Section 3: ML Suite Caffe Flow End2End w/ MNIST Dataset  
For this part of the lab, you will use a Jupyter Notebook.  
This requires launching a Jupyter Notebook server from within the container.  
Here you will train LeNet on the MNIST hand written digit dataset.  
This example was leveraged from BVLC/Caffe  
You can train using Caffe, then use ML Suite to deploy.  

1. Start the container in interactive mode, and enter it (if you are not already in it.)  
  `$ ./docker_run.sh `  
2. Start Jupyter Notebook Server in the mnist directory 
  `$ cd /opt/ml-suite/share/mnist && jupyter notebook --no-browser --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' `  
6. Navigate in a web browser to the Jupyter Notebook Server.  
  ```
  # You should use the Public DNS(IPv4) of your instance and the default port of 8888 
  # i.e. http://ec2-3-94-181-134.compute-1.amazonaws.com:8888
  ```
7. Step through the three provided notebooks  
  `LeNet Training.ipynb`  
  `LeNet Inference with CPU.ipynb`  
  `LeNet Inference with FPGA.ipynb`  
8. Stop your Jupyter Notebook Server (Don't exit the container!!!)
  `CTRL+C twice, should kill the server`

# Section 4: Real Application - Face Detection in Xilinx FPGAs
For this part of the lab, we will pretend that we are living in the real world...  
With this AWS instance, a Xilinx FPGA, and the ML Suite 1.4 container you are empowered to do many real things.  
Here we will walk through sending webcam captured frames from your laptop to a Flask RESTful server running on AWS.


1. Start the container in interactive mode, and enter it (if you are not already in it.)  
  `$ ./docker_run.sh `  
2. Start the Flask RESTful server  
  `$ cd /opt/ml-suite/share/mp_detect_clean && python app.py `
3. Set up the client side software to send webcam frames from your laptop to the RESTful server.  
  ```
  # Detailed instructions provided here:
    https://github.com/wilderfield/webcam-client
  # Go there in a web browser on your local laptop
  ``` 
