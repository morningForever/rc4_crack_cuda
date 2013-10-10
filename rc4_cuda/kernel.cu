#include "rc4.h"
/************************************************************************/
/* 
本来的思路是每次获取一个密钥，解密对应的密文，看得到的明文是否满足某个条件，
但是过程中需要的中间变量太多了，转念一想，明文和密文是异或的关系，那么已知明
文和密文异或的话就能得到密钥流的某些位置的值。这样就可以省去不少空间~~
*/
/************************************************************************/

//21-7E,totally 94 characters
#define START_CHARACTER 0x21
#define END_CHARACTER 0x7E
#define KEY (END_CHARACTER-START_CHARACTER+1)

#define MAX_KEY_LENGTH 10 //max key length
#define BLOCK_NUM 64
#define MEMEORY_PER_THREAD 266

__constant__ unsigned long long maxNum=0xFFFFFFFFFFFFFFFF;
__constant__ unsigned int maxKeyLen=MAX_KEY_LENGTH;
__constant__ unsigned int keyNum=KEY;
__constant__ unsigned int start=START_CHARACTER;
__constant__ unsigned int memory_per_thread=MEMEORY_PER_THREAD;

extern __shared__ unsigned char shared_mem[];

__device__ unsigned char* generate_key(int val,unsigned char*len)
{
	unsigned char*res=(unsigned char*)malloc(sizeof(unsigned char)*MAX_KEY_LENGTH);
	unsigned char p=MAX_KEY_LENGTH-1;
	res[p]=0;
	while (val) {
		res[p--] = (val - 1) % keyNum + start;
		val = (val - 1) / keyNum;
	}
	*len=MAX_KEY_LENGTH-1-p;
	return res+p+1;
}

__global__ void crackRc4Kernel(const unsigned char* knownKeyStream, const int known_stream_len, unsigned char*key, volatile bool *found)
{
	if(*found) return;

	int bdx=blockIdx.x, tid=threadIdx.x, keyLen=0, p=0;
	const unsigned long long keyNum_per_thread=maxNum/(BLOCK_NUM*blockDim.x)+1;

	unsigned long long val=(tid+bdx*blockDim.x)*keyNum_per_thread;
//	unsigned long long val=(tid+bdx*blockDim.x);
	unsigned long long temp;
	unsigned char res,x=0,y=0;
	for (unsigned long long i=0; i<keyNum_per_thread; val++,i++)
	{
		if(*found) return;

		if(val==0) continue;

		p = (maxKeyLen-1)+memory_per_thread*tid;
		temp=val;
		while (temp&&p>=memory_per_thread*tid) {
			shared_mem[p--] = (temp - 1) % keyNum + start;
			temp = (temp - 1) / keyNum;
		}
		keyLen=maxKeyLen+memory_per_thread*tid-p-1;

		if(*found) return;
		prepare_key(&shared_mem[p+1],keyLen,&shared_mem[maxKeyLen+memory_per_thread*tid]);

		if(*found) return;
		bool justIt=true;
		x=0,y=0;
		for(unsigned char j=0;j<4;j++){
			res=rc4_single(&x,&y,&shared_mem[maxKeyLen+memory_per_thread*tid]);
			if(knownKeyStream[j]!=res){
				justIt=false;
				break;
			}
		}
		
		if (!justIt)
		{
			continue;
		}

		*found=true;
		memcpy(key,&shared_mem[p+1],keyLen);
		key[keyLen]=0;
		__threadfence();
		asm("exit;");
		break;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t crackRc4WithCuda(unsigned char* knownKeyStream, int stream_len, unsigned char*key, bool*found)
{
	unsigned char *knownKeyStream_dev, *key_dev ;
	bool* found_dev;
	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&knownKeyStream_dev, stream_len * sizeof(unsigned char));
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
	cudaStatus = cudaMemcpy(knownKeyStream_dev, knownKeyStream, stream_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(found_dev, found, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int threadNum=prop.sharedMemPerBlock/MEMEORY_PER_THREAD;
//	threadNum=160;
	crackRc4Kernel<<<BLOCK_NUM, threadNum, prop.sharedMemPerBlock>>>(knownKeyStream_dev, stream_len, key_dev,found_dev);

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
	cudaFree(knownKeyStream_dev);
	cudaFree(key_dev);
	cudaFree(found_dev);

	return cudaStatus;
}

int main(int argc, char *argv[])
{
	unsigned char* s_box = (unsigned char*)malloc(sizeof(unsigned char)*256);
	//密钥
	unsigned char encryptKey[]="!+08";
	//明文
	unsigned char buffer[] = "Life is a chain of moments of enjoyment, not only about survivalO(∩_∩)O~";
	int buffer_len=strlen((char*)buffer);
	prepare_key(encryptKey,strlen((char*)encryptKey),s_box);
	rc4(buffer,buffer_len,s_box);
	

	/*prepare_key(encryptKey,strlen((char*)encryptKey),s_box);
	rc4(buffer,buffer_len,s_box);

	printf("%s",buffer);*/
	unsigned char knownPlainText[]="Life";
	int known_p_len=strlen((char*)knownPlainText);
	unsigned char* knownKeyStream=(unsigned char*)malloc(sizeof(unsigned char)*known_p_len);
	for (int i=0;i<known_p_len;i++)
	{
		knownKeyStream[i]=knownPlainText[i]^buffer[i];
	}

	unsigned char * key=(unsigned char*)malloc( sizeof(unsigned char) * (MAX_KEY_LENGTH+1));

	LARGE_INTEGER nFreq,nBeginTime,nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime); 

	bool found=false;
	cudaError_t cudaStatus = crackRc4WithCuda(knownKeyStream, known_p_len , key, &found);
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
		prepare_key(key,strlen((char*)encryptKey),s_box);
		rc4(buffer,buffer_len,s_box);
		printf ("\nThe clear text is:\n%s\n",buffer);
	}

	free(key);
	free(knownKeyStream);
	free(s_box);
	return 0;
}



