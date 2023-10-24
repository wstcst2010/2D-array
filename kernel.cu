/**
 * 2D TLM for CUDA
 * Simulate a network divided into N*N segments (nodes) of length dl.
 * The origin of the line matches the source impedance,
 * There is no reflection from the left side of the source. The line is
 * excited with a Gaussian voltage at the node Ein{ x, y }, and the line
 * is Terminated by a short to ground (causing equal and opposite
 * reflections at the end of the line).
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>// for setprecision
#include <fstream>
#include <ctime> //for timing
#define M_PI 3.14276    //PI
#define c 299792458     // speed of light in a vacuum
#define mu0 M_PI*4e-7   // magnetic permeability in a vacuum H/m
#define eta0 c*mu0      // wave impedance in free space 
 //Use the const function to define global variables that will not change for calling
const int NX = 100;     // number of X nodes
const int NY = 100;     // number of Y nodes
const int NT = 8192;    // number of time steps
//boundary coefficients
const int rXmin = -1;
const int rXmax = -1;
const int rYmin = -1;
const int rYmax = -1;
using namespace std;
// GPU error checking
static void HandleError(cudaError_t err,
    const char* file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
/**
* Define a structure variable that points to the storage space
* and use the structure to facilitate subsequent call variables
*/
struct dev_data {
    double* V1;      // Array for data points
    double* V2;      // Array for data points
    double* V3;      // Array for data points
    double* V4;      // Array for data points
    double* result;  // Array for voltages at the output node
    int* dev_Ein;   // input node
    int* dev_Eout;  // output node
};
// initialize arrays of 02
__global__ void initialize(dev_data dev) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;   //use auto calculate tid
    auto stride = blockDim.x * gridDim.x;               //use auto calculate stride
    while (tid < NX * NY) {
        dev.V1[tid] = 0;
        dev.V2[tid] = 0;
        dev.V3[tid] = 0;
        dev.V4[tid] = 0;
        /**
        * After each thread calculates the task on the current index, the index needs to be incremented,
        * Among them, the incremental step size is the number of threads running in the thread grid,
        * This value is equal to the number of threads in the thread block multiplied by the number of thread blocks in the thread grid,
        * ie blockDim.x * gridDim.x
        * This method is similar to the parallelism of multi-CPU or multi-core CPU, the increment of data iteration is not 1,
        * It is the number of CPUs; in GPU implementation, the number of parallel threads is generally regarded as the number of processors
        */
        tid += stride;
    }
    __syncthreads(); //Synchronize threads in a thread block, which is used to ensure
}
// apply source
__global__ void Source(dev_data dev, const double E0) {
    auto tmp = dev.dev_Ein[0] + dev.dev_Ein[1] * NX; //use auto to calculate tmp
    //Apply source voltage to the core at the input node provided
    dev.V1[tmp] = dev.V1[tmp] + E0;
    dev.V2[tmp] = dev.V2[tmp] - E0;
    dev.V3[tmp] = dev.V3[tmp] - E0;
    dev.V4[tmp] = dev.V4[tmp] + E0;
    __syncthreads(); //Synchronize threads in a thread block, which is used to ensure
}
//scatter
__global__ void Scatter(dev_data dev) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;   //use auto calculate tid
    auto stride = blockDim.x * gridDim.x;               //use auto calculate stride
    while (tid < NX * NY) {
        double I = ((dev.V1[tid] + dev.V4[tid] - dev.V2[tid] - dev.V3[tid]) / 2); // Calculate coefficient for double ,if use auto ,the final value will be wrong
        //v1
        // auto Z = eta0 / sqrt(2.);
        // double  I = (2 * (dev.V1[tid] + dev.V4[tid] - dev.V2[tid] - dev.V3[tid]) / (4 * Z));
        dev.V1[tid] = dev.V1[tid] - I; //port1
        dev.V2[tid] = dev.V2[tid] + I; //port2
        dev.V3[tid] = dev.V3[tid] + I; //port3
        dev.V4[tid] = dev.V4[tid] - I; //port4
        tid += stride;
    }
    __syncthreads(); //Synchronize threads in a thread block, which is used to ensure
}
// connect
__global__ void Connect(dev_data dev) { // boundary variables
    double tempV = 0; //Define temporary variables, cannot use auto
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;   //use auto calculate tid
    auto stride = blockDim.x * gridDim.x;               //use auto calculate stride
    //Calculate the part where x is greater than 0
    for (auto i = (tid + NY); i < (NX * NY); i += stride) {
        tempV = dev.V2[i];
        dev.V2[i] = dev.V4[i - NY];
        dev.V4[i - NY] = tempV;
    }
    __syncthreads(); //Synchronize threads in a thread block, which is used to ensure
    //Calculate the part where y is greater than 0
    for (auto i = tid + 1; i < (NX * NY); i += stride) { // Loop only through nodes where Y > 0
        if (i % NY != 0) {//y can not equal to 0
            tempV = dev.V1[i];
            dev.V1[i] = dev.V3[i - 1];
            dev.V3[i - 1] = tempV;
        }
    }
    __syncthreads(); //Synchronize threads in a thread block, which is used to ensure    
}
//boundary
__global__ void Boundary(dev_data dev) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;   //use auto calculate tid
    auto stride = blockDim.x * gridDim.x;               //use auto calculate stride
    //0*NY+Y equal to Y so Y=tid*NY
    while (tid < NX) {
        dev.V3[tid * NY + NY - 1] = rYmax * dev.V3[tid * NY + NY - 1];
        dev.V1[tid * NY] = rYmin * dev.V1[tid * NY];
        tid += stride;
    }
    __syncthreads(); //Synchronize threads in a thread block, which is used to ensure
    //0*NY+Y equal to Y so Y = tid*NY
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < NY) {
        dev.V4[(NX - 1) * NY + tid] = rXmax * dev.V4[(NX - 1) * NY + tid];
        dev.V2[tid] = rXmin * dev.V2[tid];
        tid += stride;
    }
    __syncthreads(); //Synchronize threads in a thread block, which is used to ensure
}
//result
__global__ void Result(dev_data dev, const int num) {
    dev.result[num] = dev.V2[dev.dev_Eout[0] * NY + dev.dev_Eout[1]] + dev.V4[dev.dev_Eout[0] * NY + dev.dev_Eout[1]];  //out
}
int main() {
    double dl = 1;                                                              // set node line segment length in metres
    double dt = dl / (sqrt(2.) * c);                                            // set time step duration
    //Use properties to get the number of GPU threads and block
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);                                    //interrogation  GPU 
    int threadsPerBlock = properties.maxThreadsPerBlock;                        //get threads
    int blocksPerGrid = ((NX * NY) + threadsPerBlock - 1) / threadsPerBlock;    //get block
    //Define arrays, the meaning is the same as the structure variable
    double* V1;
    double* V2;
    double* V3;
    double* V4;
    double* result;
    int* dev_Ein;
    int* dev_Eout;
    double* dev_out = new double[NT]();                                         // define output array
    std::clock_t start = std::clock();                                          // start clock
    double width = 20 * dt * sqrt(2.);                                          // gaussian width 
    double delay = 100 * dt * sqrt(2.);                                         // set time delay before starting excitation
    int Ein[] = { 10,10 };                                                      // input position
    int Eout[] = { 15,15 };                                                     // read position
    //std::ofstream output("GPU.csv");
    ofstream output("output1.out");
    cudaDeviceSynchronize();                                                    //  Initialise GPU
    //allocate memory on device
    HANDLE_ERROR(cudaMalloc((void**)&V1, (NX * NY * sizeof(double))));
    HANDLE_ERROR(cudaMalloc((void**)&V2, (NX * NY * sizeof(double))));
    HANDLE_ERROR(cudaMalloc((void**)&V3, (NX * NY * sizeof(double))));
    HANDLE_ERROR(cudaMalloc((void**)&V4, (NX * NY * sizeof(double))));
    HANDLE_ERROR(cudaMalloc((void**)&result, (NT * sizeof(double))));
    HANDLE_ERROR(cudaMalloc((void**)&dev_Ein, sizeof(int) * 2));
    HANDLE_ERROR(cudaMalloc((void**)&dev_Eout, sizeof(int) * 2));
    //copy memory areas from host to device
    HANDLE_ERROR(cudaMemcpy(dev_Ein, Ein, sizeof(int) * 2, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_Eout, Eout, sizeof(int) * 2, cudaMemcpyHostToDevice));
    dev_data dev_Data{ V1, V2, V3, V4 ,result,dev_Ein,dev_Eout };               //Structure variables are easy to call
    cudaDeviceSynchronize();                                                    // Synchronise device 
// initialise host arraysarrays to 0
    initialize << < blocksPerGrid, threadsPerBlock >> > (dev_Data);
    cudaDeviceSynchronize();                                                    // Synchronise device
// Start of TLM algorithm
//
// loop over total time NT in steps of dt
    for (int n = 0; n < NT; n++) {
        double E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width)); // calculate value of gaussian ecitation voltage at time point
/**
* In general, scatter and join are launched as two separate kernels to ensure
* that scatter is done before join. Since different steps require different numbers of cores,
* each step is distinguished here for easy research
** If the whole step is divided into two steps, the calculation time should be much faster
*/
// excitation function
        Source << <1, 1 >> > (dev_Data, E0);
        cudaDeviceSynchronize();
        // tlm scatter process
        Scatter << <blocksPerGrid, threadsPerBlock >> > (dev_Data);
        cudaDeviceSynchronize();
        //Kernel for propagating scattered pulses and applying boundary conditions
        // 
        // tlm connect process
        Connect << <blocksPerGrid, threadsPerBlock >> > (dev_Data);
        cudaDeviceSynchronize();
        // tlm boundary process
        Boundary << <blocksPerGrid, threadsPerBlock >> > (dev_Data);
        cudaDeviceSynchronize();
        // tlm calculate result process
        Result << <1, 1 >> > (dev_Data, n);
        //per 100 output n
        if (n % 100 == 0)
            cout << n << endl;
    }
    // End of TLM algorithm
    HANDLE_ERROR(cudaMemcpy(dev_out, result, (NT * sizeof(double)), cudaMemcpyDeviceToHost)); // copy array of measured voltages from device
    // write measured voltages to file
    for (int i = 0; i < NT; ++i) {
        output << i * dt << " " << dev_out[i] << endl;                                        //output the result to file
    }
    // free memory allocated on the GPU
    HANDLE_ERROR(cudaFree(V1));
    HANDLE_ERROR(cudaFree(V2));
    HANDLE_ERROR(cudaFree(V3));
    HANDLE_ERROR(cudaFree(V4));
    HANDLE_ERROR(cudaFree(result));
    HANDLE_ERROR(cudaFree(dev_Ein));
    HANDLE_ERROR(cudaFree(dev_Eout));
    output.close();
    cout << "Done";
    std::cout << ((std::clock() - start) / (double)CLOCKS_PER_SEC) << '\n';                 //output time
    cin.get();
}