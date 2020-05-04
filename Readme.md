# TrtMatrix
`TrtMatrix`是一个基于`TensorRT`的Demo项目，其中包含了`TrtTransformer`和`TrtExecutor`两个子项目工程。

依赖TensorRT 7.x。

## TrtTransformer
这是一个支持将其它深度学习框架模型转换为TensorRT模型的工具。当前支持ONNX模型。

## TrtExecutor
执行TensorRT模型推理任务，main.cpp当前实现了一个音频降噪任务处理，原始模型来自于[microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)，基于`DNS-Challenge/NSNet-baseline`下模型与前后处理Python代码移植。

## ThirdParty/SubModule
本项目依赖以下第三方工程，以Git Submodule形式引入。第三方项目工程遵守其原始开源分发协议。

* [AudioFFT](https://github.com/HiFi-LoFi/AudioFFT)。用于计算Real FFT相关。
* [libsndfile](https://github.com/erikd/libsndfile)。用于读写wav音频文件。
* [NumCpp](https://github.com/dpilger26/NumCpp)。`NumPy`部分函数的cpp实现库，用于快速移植。

## License
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

Copyright (c) 2020 Mematrix

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files(the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
