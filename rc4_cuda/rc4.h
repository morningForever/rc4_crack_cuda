/* rc4.h */ 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <Windows.h>

//21-7E,totally 94 characters
#define START_CHARACTER 0x21
#define END_CHARACTER 0x7E
#define KEY (END_CHARACTER-START_CHARACTER+1)

#define MAX_KEY_LENGTH 10 //max key length
#define BLOCK_NUM 2
//空间其实只要10个就足够了，取20的原因主要是为了避免bank conflicts
#define MEMEORY_PER_THREAD 20
#define MAX_THREAD_NUM 256
#define STATE_LEN	256

__constant__ unsigned long long maxNum=0xFFFFFFFFFFFFFFFF;
__constant__ unsigned int maxKeyLen=MAX_KEY_LENGTH;
__constant__ unsigned int keyNum=KEY;
__constant__ unsigned int start=START_CHARACTER;
__constant__ unsigned int memory_per_thread=MEMEORY_PER_THREAD;

extern __shared__ unsigned char shared_mem[];

__device__ __host__ unsigned char rc4_single(unsigned char*x, unsigned char * y, unsigned char *s_box);
__device__ __host__ static void swap_byte(unsigned char *a, unsigned char *b);
__device__ bool device_isKeyRight(const unsigned char *known_stream, int known_len,unsigned char *validateKey,int key_len);
__device__ __host__ unsigned char rc4_single(unsigned char*x, unsigned char * y, unsigned char *s_box);
void prepare_key(unsigned char *key_data_ptr, int key_data_len,unsigned char *s_box);
void rc4(unsigned char *buffer_ptr, int buffer_len, unsigned char *s_box);
/************************************************************************/
/* the data type is unsigned char,so the %256 is no necessary           */
/************************************************************************/

/**
 * \brief swap two bytes
 */
__device__ __host__ static void swap_byte(unsigned char *a, unsigned char *b) 
{ 
	unsigned char swapByte;  

	swapByte = *a;  
	*a = *b;      
	*b = swapByte; 
}

__device__ bool device_isKeyRight(const unsigned char *known_stream, int known_len,unsigned char *validateKey,int key_len,volatile bool* found) 
{ 
	//KSA
	unsigned char state[STATE_LEN];
	unsigned char index1=0, index2=0;
	short counter=0;    

    if(*found) asm("exit;");   

	for(counter = 0; counter < STATE_LEN; counter++)          
		state[counter] = counter;   

	if(*found) asm("exit;");

	for(counter = 0; counter < STATE_LEN; counter++)      
	{             
		index2 = (validateKey[index1] + state[counter] + index2);            
		swap_byte(&state[counter], &state[index2]);
		index1 = (index1 + 1) % key_len;  
	} 

	if(*found) asm("exit;");

	//PRGA
	index1=0, index2=0, counter=0; 
	for (;counter<known_len;counter++)
	{
		if(known_stream[counter]!=rc4_single(&index1,&index2,state))
			return false;
	}

	if(*found) asm("exit;");

	return true;
} 
/**
 * \brief rc4 encryption and decryption function
 * 
 * \param buffer_ptr,the data string to encryption 
 * \param buffer_len,the data length
 * \param key,rc4's s-box and the two key pointers,this was used to encryption the data
 *
 * \return void
**/
__device__ __host__ unsigned char rc4_single(unsigned char*x, unsigned char * y, unsigned char *s_box) 
{  
	unsigned char* state, xorIndex; 

	state = &s_box[0];    

	*x = (*x + 1);                
	*y = (state[*x] + *y);            
	swap_byte(&state[*x], &state[*y]);                  

	xorIndex = (state[*x] + state[*y]);            

	return  state[xorIndex];        
} 

/**
 * \brief rc4 s-box init
 * 
 * \param key_data_ptr,the encryption key
 * \param key_data_len,the encryption key length,less than 256
 * \param key,rc4's s-box and the key two pointers
 *
 * \return void
**/
void prepare_key(unsigned char *key_data_ptr, int key_data_len,unsigned char *s_box) 
{ 
	unsigned char index1=0, index2=0, * state; 
	short counter;    

	state = &s_box[0];        
	for(counter = 0; counter < STATE_LEN; counter++)          
		state[counter] = counter;   
	for(counter = 0; counter < STATE_LEN; counter++)      
	{             
		index2 = (key_data_ptr[index1] + state[counter] + index2);            
		swap_byte(&state[counter], &state[index2]);          

		index1 = (index1 + 1) % key_data_len;  
	}      
} 

void rc4(unsigned char *buffer_ptr, int buffer_len, unsigned char *s_box) 
{  
	unsigned char x=0, y=0, * state;
	short counter; 

	state = &s_box[0];        
	for(counter = 0; counter < buffer_len; counter ++)
	{  
		buffer_ptr[counter] ^= rc4_single(&x,&y,state);        
	}            
} 