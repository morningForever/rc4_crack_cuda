/* rc4.h */ 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Windows.h>
typedef struct rc4_key 
{      
	unsigned char state[256];      
	unsigned char x;      
	unsigned char y; 
} rc4_key; 

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

/**
 * \brief rc4 s-box init
 * 
 * \param key_data_ptr,the encryption key
 * \param key_data_len,the encryption key length,less than 256
 * \param key,rc4's s-box and the key two pointers
 *
 * \return void
**/
__device__ __host__ void prepare_key(unsigned char *key_data_ptr, int key_data_len,rc4_key *key) 
{ 
	unsigned char index1; 
	unsigned char index2; 
	unsigned char* state; 
	short counter;    

	state = &key->state[0];        
	for(counter = 0; counter < 256; counter++)          
		state[counter] = counter;            
	key->x = 0;    
	key->y = 0;    
	index1 = 0;    
	index2 = 0;          
	for(counter = 0; counter < 256; counter++)      
	{            
	//	index2 = (key_data_ptr[index1] + state[counter] + index2) % 256;    
		index2 = (key_data_ptr[index1] + state[counter] + index2);            
		swap_byte(&state[counter], &state[index2]);          

		index1 = (index1 + 1) % key_data_len;  
	}      
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
__device__ __host__ void rc4(unsigned char *buffer_ptr, int buffer_len, rc4_key *key) 
{  
	unsigned char x; 
	unsigned char y; 
	unsigned char* state; 
	unsigned char xorIndex; 
	short counter;          

	x = key->x;    
	y = key->y;    

	state = &key->state[0];        
	for(counter = 0; counter < buffer_len; counter ++)      
	{            
	//	x = (x + 1) % 256;                
	//	y = (state[x] + y) % 256;    
		x = (x + 1);                
		y = (state[x] + y);            
		swap_byte(&state[x], &state[y]);                  

	//	xorIndex = (state[x] + state[y]) % 256;  
		xorIndex = (state[x] + state[y]);            

		buffer_ptr[counter] ^= state[xorIndex];        
	}            
	key->x = x;    
	key->y = y; 
} 
