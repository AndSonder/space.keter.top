# 编译和执行

:::tip

本文中将结合前面的代码，介绍如何编译和执行CUDA程序。并写一个在GPU上执行的向量加法程序。

:::

## 基于GPU的向量加法

首先第一步先引入需要用到的头文件

```c
#include <stdio.h>
#include <cuda_runtime.h>
```

然后为什么我们需要一个错误检查的宏，这个宏可以帮助我们检查CUDA API调用是否成功，如果失败，它会打印出错误信息。这个宏在上一篇文章中已经介绍过了，这里就不再赘述了。

```c
#define CHECK(call)
{
    const cudaError_t error = call;
    if (error != cudaSuccess)
    {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
```

接下来我们需要定义一个核函数，这个核函数用来实现向量的加法，这个核函数的实现非常简单，就是将两个向量对应位置的元素相加，然后将结果保存到第三个向量中。

```c
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
```

为了验证我们的核函数是否正确，我们需要在主机端定义一个函数，这个函数用来在主机端实现向量的加法，然后将结果保存到第三个向量中。

```c
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}
```

下面我们写一个辅助函数，这个函数用来初始化向量，这里我们使用随机数来初始化向量。

```c
void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    // set seed
    srand((unsigned int)time(&t));

    // initialize the input data on the host side
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}
```

然后我们写一个辅助函数，这个函数用来检查两个向量是否相等，如果相等，那么就说明我们的核函数实现正确。

```c
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}
```

下面我们就可以写主函数了，首先我们需要定义一些变量，这些变量用来保存向量的长度，以及向量的大小。

```c
int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
```

然后我们需要在设备端分配内存，这里我们使用`cudaMalloc`函数来分配内存，这个函数的第一个参数是一个指针，这个指针指向的是我们需要分配的内存的地址，第二个参数是我们需要分配的内存的大小。

```c
    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);
```

然后我们需要将数据从主机端拷贝到设备端，这里我们使用`cudaMemcpy`函数来拷贝数据，这个函数的第一个参数是目标地址，第二个参数是源地址，第三个参数是拷贝的数据的大小，第四个参数是拷贝的方向，这里我们需要将数据从主机端拷贝到设备端，所以我们需要将第四个参数设置为`cudaMemcpyHostToDevice`。

```c
    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
```

然后我们需要定义一个变量，这个变量用来保存我们的核函数的执行配置，这个变量是一个结构体，这个结构体有三个成员，分别是`dim3`类型的`block`和`grid`，以及`unsigned int`类型的`sharedMem`，这里我们只需要使用`block`和`grid`这两个成员，`block`用来指定每个线程块中有多少个线程，`grid`用来指定有多少个线程块。

```c
    // set up execution configuration
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
```

这里我们将`block`设置为`1024`，这样我们就可以保证每个线程块中有`1024`个线程，然后我们将`grid`设置为`nElem`除以`block.x`，这样我们就可以保证有足够的线程块来处理所有的数据。

然后我们就可以调用我们的核函数了，这里我们需要传入三个参数，第一个参数是一个指针，这个指针指向的是第一个向量，第二个参数是一个指针，这个指针指向的是第二个向量，第三个参数是一个指针，这个指针指向的是第三个向量，这里我们需要将第三个向量的结果保存到这个指针指向的内存中，最后一个参数是一个`unsigned int`类型的变量，这个变量用来指定我们的向量的长度。

```c
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
```

然后我们需要将数据从设备端拷贝到主机端，这里我们需要将数据从设备端拷贝到主机端，所以我们需要将第四个参数设置为`cudaMemcpyDeviceToHost`。

```c
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
```

然后我们需要调用我们的CPU函数来计算结果，这里我们需要传入三个参数，第一个参数是一个指针，这个指针指向的是第一个向量，第二个参数是一个指针，这个指针指向的是第二个向量，第三个参数是一个指针，这个指针指向的是第三个向量，这里我们需要将第三个向量的结果保存到这个指针指向的内存中，最后一个参数是一个`unsigned int`类型的变量，这个变量用来指定我们的向量的长度。

```c
    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
```

然后我们需要检查我们的结果是否正确，这里我们需要传入三个参数，第一个参数是一个指针，这个指针指向的是第一个向量，第二个参数是一个指针，这个指针指向的是第二个向量，第三个参数是一个`unsigned int`类型的变量，这个变量用来指定我们的向量的长度。

```c
    checkResult(hostRef, gpuRef, nElem);
```

最后我们需要释放我们在设备端分配的内存，这里我们需要调用`cudaFree`函数来释放我们在设备端分配的内存，这个函数的参数是一个指针，这个指针指向的是我们需要释放的内存的地址。

```c
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();
```

最后我们需要调用`cudaDeviceReset`函数来重置我们的设备，这个函数没有参数，这个函数的作用是重置当前设备上的所有状态，这个函数一般在程序退出之前调用。

## 编译和运行

我们可以使用以下命令来编译和运行这个程序。

```shell
$ nvcc -o vectorAdd vectorAdd.cu
$ ./vectorAdd
Starting...
Using Device 0: NVIDIA GeForce RTX 3090
Vector size 16777216
Arrays match.
```

## 完整代码

```c
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                           \ 
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
        exit(1);                                                                \
    }                                                                           \ 
}                                                                               \ 

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    // set seed
    srand((unsigned int)time(&t));

    // initialize the input data on the host side
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();

    return 0;
}
```








## 参考资料

1. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
2. [【CUDA 基础】2.0 CUDA编程模型概述（二）](https://face2ai.com/CUDA-F-2-1-CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B02/)



