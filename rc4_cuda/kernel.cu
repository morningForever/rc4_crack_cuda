#include "rc4.h"
/************************************************************************/
/* 
本来的思路是每次获取一个密钥，解密对应的密文，看得到的明文是否满足某个条件，
但是过程中需要的中间变量太多了，转念一想，明文和密文是异或的关系，那么已知明
文和密文异或的话就能得到密钥流的某些位置的值。这样就可以省去不少空间~~
*/
/************************************************************************/

__device__ unsigned char* genKey(unsigned char*res,unsigned long long val,int*key_len)
{
	char p=maxKeyLen-1;
	while (val&&p>=0) {
		res[p--] = (val - 1) % keyNum + start;
		val = (val - 1) / keyNum;
	}
	*key_len=(maxKeyLen-p-1);
	return res+p+1;
}

__global__ void crackRc4Kernel(unsigned char*key, volatile bool *found)
{
	if(*found) asm("exit;");

	int bdx=blockIdx.x, tid=threadIdx.x, keyLen=0;
	const unsigned long long totalThreadNum=gridDim.x*blockDim.x;
	const unsigned long long keyNum_per_thread=maxNum/totalThreadNum;
//	unsigned long long val=(tid+bdx*blockDim.x)*keyNum_per_thread;
	unsigned long long val=(tid+bdx*blockDim.x);
	bool justIt=true;
	for (unsigned long long i=0; i<=keyNum_per_thread; val+=totalThreadNum,i++)
	{
		//找到的话退出
		if(*found) asm("exit;");
		if(val==0) continue;

		//vKey是share_memory的一个指针
		unsigned char*vKey=genKey((shared_mem+memory_per_thread*tid),val,&keyLen);

		//找到的话退出
		if(*found) asm("exit;");

		justIt=device_isKeyRight(vKey,keyLen,found);

		//找到的话退出
		if(*found) asm("exit;");

		//当前密钥不是所求
		if (!justIt) continue;

		//找到匹配密钥，写到Host，保存数据,修改found,退出程序
		*found=true;
		memcpy(key,vKey,keyLen);
		key[keyLen]=0;
		__threadfence();
		asm("exit;");
		break;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t crackRc4WithCuda(unsigned char* knownKeyStream_host, int knownStreamLen_host, unsigned char*key, bool*found)
{
	unsigned char *key_dev ;
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

	//检验是否找到密钥变量
	cudaStatus = cudaMemcpy(found_dev, found, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//复制常量内存
	cudaStatus = cudaMemcpyToSymbol(knowStream_device, knownKeyStream_host,sizeof(unsigned char)*knownStreamLen_host);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpyToSymbol((const void *)&knownStreamLen_device,(const void *)&knownStreamLen_host,sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int threadNum=floor((double)(prop.sharedMemPerBlock/MEMEORY_PER_THREAD)),share_memory=prop.sharedMemPerBlock;
	if(threadNum>MAX_THREAD_NUM){
		threadNum=MAX_THREAD_NUM;
		share_memory=threadNum*MEMEORY_PER_THREAD;
	}
	crackRc4Kernel<<<BLOCK_NUM, threadNum, share_memory>>>(key_dev,found_dev);

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
	cudaFree(key_dev);
	cudaFree(found_dev);

	return cudaStatus;
}

int main(int argc, char *argv[])
{
//	printf("%c",0x7d);
	unsigned char* s_box = (unsigned char*)malloc(sizeof(unsigned char)*256);
	//密钥
	unsigned char encryptKey[]="!!!}";
	//明文
	unsigned char buffer[] = "Life is a chain of moments of enjoyment, not only about survivalO(∩_∩)O~";
	int buffer_len=strlen((char*)buffer);
	prepare_key(encryptKey,strlen((char*)encryptKey),s_box);
	rc4(buffer,buffer_len,s_box);	

	unsigned char knownPlainText[]="Life";
	int known_p_len=strlen((char*)knownPlainText);
	unsigned char* knownKeyStream=(unsigned char*)malloc(sizeof(unsigned char)*known_p_len);
	for (int i=0;i<known_p_len;i++)
	{
		knownKeyStream[i]=knownPlainText[i]^buffer[i];
	}

	unsigned char * key=(unsigned char*)malloc( sizeof(unsigned char) * (MAX_KEY_LENGTH+1));

	cudaEvent_t start,stop;
	cudaError_t cudaStatus=cudaEventCreate(&start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(start) failed!");
		return 1;
	}
	cudaStatus=cudaEventCreate(&stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(stop) failed!");
		return 1;
	}

	cudaStatus=cudaEventRecord(start,0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(start) failed!");
		return 1;
	}

	bool found=false;
	cudaStatus = crackRc4WithCuda(knownKeyStream, known_p_len , key, &found);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus=cudaEventRecord(stop,0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(stop) failed!");
		return 1;
	}

	cudaStatus=cudaEventSynchronize(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventSynchronize failed!");
		return 1;
	}
	float useTime;
	cudaStatus=cudaEventElapsedTime(&useTime,start,stop);
	useTime/=1000;
	printf("The time we used was:%fs\n",useTime);
	if (found)
	{
		printf("The right key has been found.The right key is:%s\n",key);
		prepare_key(key,strlen((char*)key),s_box);
		rc4(buffer,buffer_len,s_box);
		printf ("\nThe clear text is:\n%s\n",buffer);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(key);
	free(knownKeyStream);
	free(s_box);
	cudaThreadExit();
	return 0;
}



