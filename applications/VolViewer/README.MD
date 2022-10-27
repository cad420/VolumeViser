## VolViewer
实时体渲染以及切片渲染 支持同个主机同步并行渲染
使用ImGUI模式


由于OpenGL只能使用单独的一个显卡，因此如果在多显卡环境下，需要将CUDA绘制的结果先传输到CPU，再上传到OpenGL的纹理中，
这样子显然有时间损耗
因此窗口的创建需要用到Vulkan，直接将CUDA绘制的结果，注入到Vulkan的Image中，只需要一次GPU上的拷贝
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/vulkanImageCUDA/vulkanImageCUDA.cu
