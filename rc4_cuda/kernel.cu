#include "rc4.h"

//21-7E,totally 94 characters
#define START_CHARACTER 0x21
#define END_CHARACTER 0x7E
#define KEY (END_CHARACTER-START_CHARACTER+1)

#define MAX_KEY_LENGTH 10 //max key length
#define THREAD_NUM 256
#define BLOCK_NUM 32
#define KEY_COUNT_PER_THREAD 0x100000000000000/BLOCK_NUM

__constant__ unsigned long long maxNum=0xFFFFFFFFFFFFFFFF;
__constant__ unsigned int threadNum=THREAD_NUM;
__constant__ unsigned int blockNum=BLOCK_NUM;
__constant__ unsigned int maxKeyLen=MAX_KEY_LENGTH;
__constant__ unsigned long long keyCount_per_thread=KEY_COUNT_PER_THREAD;
__constant__ unsigned int keyNum=KEY;
__constant__ unsigned int start=START_CHARACTER;

__global__ void crackRc4Kernel(const unsigned char* buffer,unsigned char* bufferCpoy_dev, const int buf_len, unsigned char*key, volatile bool *found,unsigned char *rs,rc4_key*rc4_k)
{
	if(*found) return;

	int bdx=blockIdx.x, tid=threadIdx.x, keyLen=0, p=0;

	unsigned long long val=(tid+bdx*threadNum)*keyCount_per_thread, i=0;
	unsigned long long temp;
	for (unsigned long long i=0; i<keyCount_per_thread; val++,i++)
	{
		if(*found) return;
		if(val==0) continue;

		memcpy(bufferCpoy_dev,buffer,buf_len*sizeof(unsigned char));

		rs[maxKeyLen]=0;
		p = maxKeyLen-1;
		temp=val;
		while (temp&&p>=0) {
			rs[p--] = (temp - 1) % keyNum + start;
			temp = (temp - 1) / keyNum;
		}
		keyLen=maxKeyLen-p-1;

		prepare_key(&rs[p+1],keyLen,(rc4_key*)rc4_k);
		rc4(bufferCpoy_dev,buf_len,(rc4_key*)rc4_k);

		if(bufferCpoy_dev[0]=='L'&&bufferCpoy_dev[1]=='i'&&bufferCpoy_dev[2]=='f'&&bufferCpoy_dev[3]=='e')
		{
			*found=true;
			memcpy(key,&rs[p+1],keyLen+1);
			__threadfence();
			asm("trap;");
			break;
		}
	}
}

bool InitCUDA(void)
{
	int count = 0;
	int i = 0;
	cudaGetDeviceCount(&count); //看看有多少个设备?
	if(count == 0)   //哈哈~~没有设备.
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	cudaDeviceProp prop;
	for(i = 0; i < count; i++)  //逐个列出设备属性:
	{
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if(prop.major >= 1)
			{
				break;
			}
		}
	}
	if(i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaDeviceProp sDevProp = prop;
	printf( "%d \n", i);
	printf( "Device name: %s\n", sDevProp.name );
	printf( "Device memory: %d\n", sDevProp.totalGlobalMem );
	printf( "Memory per-block: %d\n", sDevProp.sharedMemPerBlock );
	printf( "Register per-block: %d\n", sDevProp.regsPerBlock );
	printf( "Warp size: %d\n", sDevProp.warpSize );
	printf( "Memory pitch: %d\n", sDevProp.memPitch );
	printf( "Constant Memory: %d\n", sDevProp.totalConstMem );
	printf( "Max thread per-block: %d\n", sDevProp.maxThreadsPerBlock );
	printf( "Max shared memory: %d\n", sDevProp.sharedMemPerBlock );
	printf( "Max thread dim: ( %d, %d, %d )\n", sDevProp.maxThreadsDim[0],
		sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2] );
	printf( "Max grid size: ( %d, %d, %d )\n", sDevProp.maxGridSize[0],  
		sDevProp.maxGridSize[1], sDevProp.maxGridSize[2] );
	printf( "Ver: %d.%d\n", sDevProp.major, sDevProp.minor );
	printf( "Clock: %d\n", sDevProp.clockRate );
	printf( "textureAlignment: %d\n", sDevProp.textureAlignment );
	cudaSetDevice(i);
	printf("\n CUDA initialized.\n");
	return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t crackRc4WithCuda(unsigned char* buffer, int buf_len, unsigned char*key, bool*found)
{
//	InitCUDA();
	unsigned char *buffer_dev, *key_dev, *bufferCpoy_dev;
	bool* found_dev;
	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&buffer_dev, buf_len * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}	
	
	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&bufferCpoy_dev, buf_len * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&key_dev, (MAX_KEY_LENGTH+1) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&found_dev, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(bufferCpoy_dev, buffer, buf_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(key_dev, key, (MAX_KEY_LENGTH+1) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(found_dev, found, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	unsigned char *rs_dev ;
	rc4_key* rc4_k_dev;

	cudaStatus = cudaMalloc((void**)&rs_dev, sizeof(unsigned char)*(MAX_KEY_LENGTH+1));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&rc4_k_dev, sizeof(rc4_key));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	// Launch a kernel on the GPU with one thread for each element.
	//crackRc4Kernel<<<BLOCK_NUM, THREAD_NUM>>>(buffer_dev,bufferCpoy_dev, buf_len, key_dev,found_dev,rs_dev,rc4_k_dev);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(buffer, buffer_dev, buf_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(key, key_dev, (MAX_KEY_LENGTH+1) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(found, found_dev,  sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(buffer_dev);
	cudaFree(key_dev);
	cudaFree(found_dev);
	cudaFree(bufferCpoy_dev);
	cudaFree(rc4_k_dev);
	cudaFree(rs_dev);

	return cudaStatus;
}

int main(int argc, char *argv[])
{
	rc4_key* rc4_k;
	rc4_k = (rc4_key*)malloc(sizeof(rc4_key));
	//密钥
	unsigned char encryptKey[]="!";
	//明文
	unsigned char buffer[] = "Life is a chain of moments of enjoyment, not only about survivalO(∩_∩)O~";
	int buffer_len=strlen((char*)buffer);
	prepare_key(encryptKey,strlen((char*)encryptKey),rc4_k);
	rc4(buffer,buffer_len,rc4_k);
	
	unsigned char *encryptData=(unsigned char *)strdup((char*)buffer);

	unsigned char * key=(unsigned char*)malloc( sizeof(unsigned char) * (MAX_KEY_LENGTH+1));

	LARGE_INTEGER nFreq,nBeginTime,nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime); 

	bool found=false;
	cudaError_t cudaStatus = crackRc4WithCuda(encryptData, buffer_len+1 , key, &found);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	QueryPerformanceCounter(&nEndTime);
	float time=(float)(nEndTime.QuadPart-nBeginTime.QuadPart)/((float)nFreq.QuadPart);
	printf("The time we used was:%fs\n",time);
	if (found)
	{
		printf("The right key has been found.The right key is:%s\n",key);
		prepare_key(key,strlen((char*)encryptKey),rc4_k);
		rc4(buffer,buffer_len,rc4_k);
		printf ("\nThe clear text is:\n%s\n",buffer);
	}

	free(key);
	free(encryptData);
	free(rc4_k);
	return 0;
}



