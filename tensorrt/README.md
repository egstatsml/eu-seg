## Deploy with Tensorrt 

These instructions have been modified to reflect how to compile with TensorRT using probabilistic modules within this repo.

### NOTE
Currently only supports ```--aux-mode eval-bayes-prob```. This is because the current compilation code in [segment.c](./segment.c) is hard coded for two outputs (mean and variance). I may update this to make this dynamic with a command-line argument.

### Pytorch for Jetson devices
For Jetson devices, the unofficial installation instructions [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) where useful for me.

### Export using Onnx

Firstly, We should export our trained model to onnx model:  

``` shell
python tools/export_onnx.py --config configs/bayes_bisenetv2_city.py --weight-path ./res/bisenetv2_bayes_model_final.pth --outpath ./model.onnx --aux-mode eval_bayes_prob
```


### Compiling using C++


#### Build with source code
Just use the standard cmake build method:  
``` shell
mkdir -p tensorrt/build
cd tensorrt/build
cmake ..
make
```
This would generate a `./segment` in the `tensorrt/build` directory.


#### Convert onnx to tensorrt model
If you can successfully compile the source code, you can parse the onnx model to tensorrt model with one of the following commands.   
For fp32, command is:
``` shell
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt
```
If your gpu support acceleration with fp16 inferenece, you can add a `--fp16` option to in this step:  
``` shell
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --fp16
```
Building an int8 engine is also supported. Firstly, you should make sure your gpu support int8 inference, or you model will not be faster than fp16/fp32. Then you should prepare certain amount of images for int8 calibration. In this example, I use train set of cityscapes for calibration. The command is like this:  
```
$ calibrate_int8 # delete this if exists
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --int8 /path/to/BiSeNet/datasets/cityscapes /path/to/BiSeNet/datasets/cityscapes/train.txt
```
With the above commands, we will have an tensorrt engine named `saved_model.trt` generated.  

Note that I use the simplest method to parse the command line args, so please do **Not** change the order of the args in above command.  


#### 4. Infer with one single image
Run inference like this:   
``` shell
$ ./segment run /path/to/saved_model.trt /path/to/input/image.jpg /path/to/saved_img.jpg
```


#### 5. Test speed  
The speed depends on the specific gpu platform you are working on, you can test the fps on your gpu like this:  
``` shell
$ ./segment test /path/to/saved_model.trt
```


#### 6. Tips:  

These tips where provided from original repo.

1. ~Since tensorrt 7.0.0 cannot parse well the `bilinear interpolation` op exported from pytorch, I replace them with pytorch `nn.PixelShuffle`, which would bring some performance overhead(more flops and parameters), and make inference a bit slower. Also due to the `nn.PixelShuffle` op, you **must** export the onnx model with input size to be *n* times of 32.~   
If you are using 7.2.3.4 or newer versions, you should not have problem with `interpolate` anymore.

2. ~There would be some problem for tensorrt 7.0.0 to parse the `nn.AvgPool2d` op from pytorch with onnx opset11. So I use opset10 to export the model.~  
Likewise, you do not need to worry about this anymore with version newer than 7.2.3.4.
