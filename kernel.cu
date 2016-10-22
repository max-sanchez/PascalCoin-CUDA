
#include "cuda_runtime.h"
#include <cstdint>
#include "device_launch_parameters.h"
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <ctime>

#ifdef __INTELLISENSE__
#define __launch_bounds__(blocksize)
#endif

cudaStream_t cudastream;

uint32_t *blockHeadermobj = nullptr;
uint32_t *midStatemobj    = nullptr;
uint32_t *nonceOutmobj    = nullptr;

__device__ __forceinline__ uint32_t
ror(const uint32_t a, const unsigned int n)
{
#if __CUDA_ARCH__ >= 999 // Disabled
	uint32_t d;
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(a), "r"(n));
	return d;
#else
	return (a >> n) | (a << (32 - n));
#endif
}

__device__ __forceinline__ uint32_t
shr(const uint32_t a, const unsigned int n)
{
#if __CUDA_ARCH__ >= 999 // Disabled
	uint32_t d;
	asm("vshr.u32.u32.u32.clamp %0, %1, %2;" : "=r"(d) : "r"(a), "r"(n));
	return d;
#else
	return a >> n;
#endif
}

#define ROTRIGHT(a,b) ((a >> b) | (a << (32 - b)))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))
#define SIG0c(x) (ror(x,7) ^ ror(x,18) ^ ((x) >> 3))
#define SIG1c(x) (ror(x,17) ^ ror(x,19) ^ ((x) >> 10))

#define blocksize 2048
#define npt 9

static const uint32_t k[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__global__ void __launch_bounds__(blocksize, 8) nonceGrindc(uint32_t *const __restrict__ headerIn, uint32_t *const __restrict__ midstateIn, uint32_t *const __restrict__ nonceOut)
{
	static const uint32_t k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
		0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
		0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
		0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
		0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
		0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
		0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};

	// int i = (blockIdx.x * blockDim.x * blockDim.y) + threadIdx.x;
	//i *= npt;

	uint32_t buffer[16];
	uint32_t midstate[8];

	// const uint32_t id = (blockDim.x * blockIdx.x * blockIdx.x * blockIdx.x * headerIn[16] + threadIdx.x)*npt;
	const uint32_t id = (headerIn[16] << 20) | (blockDim.x * blockIdx.x + threadIdx.x)*npt;

	midstate[0] = midstateIn[0];
	midstate[1] = midstateIn[1];
	midstate[2] = midstateIn[2];
	midstate[3] = midstateIn[3];
	midstate[4] = midstateIn[4];
	midstate[5] = midstateIn[5];
	midstate[6] = midstateIn[6];
	midstate[7] = midstateIn[7];

	int j = 0;

	for (j = 0; j < 16; j++)
	{
		buffer[j] = headerIn[j];
	}

	uint32_t block[64];

	uint32_t temp1;
	uint32_t temp2;
	uint32_t S0;
	uint32_t S1;

	uint32_t h0, h1, h2, h3, h4, h5, h6, h7;

	uint32_t a, b, c, d, e, f, g, h;
	for (int n = id; n < id + npt; n++)
	{
		h0 = midstate[0];
		h1 = midstate[1];
		h2 = midstate[2];
		h3 = midstate[3];
		h4 = midstate[4];
		h5 = midstate[5];
		h6 = midstate[6];
		h7 = midstate[7];

		a = h0;
		b = h1;
		c = h2;
		d = h3;
		e = h4;
		f = h5;
		g = h6;
		h = h7;

		buffer[11] = n;
		// printf("Nonce being used: %d\n" + buffer[11]);

		for (j = 0; j < 16; j++)
		{
			block[j] = buffer[j];
		}

		for (j = 16; j < 64; j++)
		{
			block[j] = block[j - 16] + block[j - 7] + SIG1c(block[j - 2]) + SIG0c(block[j - 15]);
		}

		for (j = 0; j < 64; j++)
		{
			S1 = (ror(e, 6)) ^ (ror(e, 11)) ^ (ror(e, 25));
			temp1 = h + S1 + ((e & f) ^ ((~e) & g)) + k[j] + block[j];
			S0 = (ror(a, 2)) ^ (ror(a, 13)) ^ (ror(a, 22));
			temp2 = S0 + (((a & b) ^ (a & c) ^ (b & c)));

			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}

		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;
		h4 += e;
		h5 += f;
		h6 += g;
		h7 += h;

		block[0] = h0;
		block[1] = h1;
		block[2] = h2;
		block[3] = h3;
		block[4] = h4;
		block[5] = h5;
		block[6] = h6;
		block[7] = h7;
		block[8] = 0x80000000;
		block[9] = 0x00000000;
		block[10] = 0x00000000;
		block[11] = 0x00000000;
		block[12] = 0x00000000;
		block[13] = 0x00000000;
		block[14] = 0x00000000;
		block[15] = 0x00000100;

		h0 = a = 0x6a09e667;
		h1 = b = 0xbb67ae85;
		h2 = c = 0x3c6ef372;
		h3 = d = 0xa54ff53a;
		h4 = e = 0x510e527f;
		h5 = f = 0x9b05688c;
		h6 = g = 0x1f83d9ab;
		h7 = h = 0x5be0cd19;

		for (j = 16; j < 64; j++)
		{
			block[j] = block[j - 16] + block[j - 7] + SIG1c(block[j - 2]) + SIG0c(block[j - 15]);
		}

		for (j = 0; j < 64; j++)
		{
			S1 = (ror(e, 6)) ^ (ror(e, 11)) ^ (ror(e, 25));
			temp1 = h + S1 + ((e & f) ^ ((~e) & g)) + k[j] + block[j];
			S0 = (ror(a, 2)) ^ (ror(a, 13)) ^ (ror(a, 22));
			temp2 = S0 + (((a & b) ^ (a & c) ^ (b & c)));

			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}

		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;
		h4 += e;
		h5 += f;
		h6 += g;
		h7 += h;

		uint32_t targetX = h0 & 0xFFFFFFFF;
		uint32_t targetY = h1 & 0xF0000000;
		if (targetX == 0 && targetY == 0)
		{
			*nonceOut = n;
			/* Uncomment these for additional mining information/verbosity */
			printf("    The hash is: \n%08x %08x %08x %08x %08x %08x %08x %08x \n", h0, h1, h2, h3, h4, h5, h6, h7);
			//printf("      headerIn[16]:  %d\n", headerIn[16]);
			//printf("      blockDim.x:    %d\n", blockDim.x);
			//printf("      blockIdx.x:    %d\n", blockIdx.x);
			//printf("      threadIdx.x:   %d\n", threadIdx.x);
			//printf("      npt:           %d\n", npt);
			//printf("    And the total:\n    %08x %08x %08x %08x %08x %08x %08x %08x \n", buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7]);
			//printf("    %08x %08x %08x %08x %08x %08x %08x %08x]\n", buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14], buffer[15]);
		}
	}
}

unsigned char* hexToByteArray(const char* hexstring)
{
	size_t len = 176 * 2;
	size_t final_len = len / 2;
	unsigned char* chrs = (unsigned char*)malloc((final_len)* sizeof(*chrs));
	for (size_t i = 0, j = 0; j<final_len; i += 2, j++)
		chrs[j] = (hexstring[i] % 32 + 9) % 25 * 16 + (hexstring[i + 1] % 32 + 9) % 25;
	return chrs;
}

#define headerSize 176

// Only used for determining hashrate, and it's this method's fault that the hashrate sometimes shows as negative (this "rolls over" since nothing over the hour is used in creating the relative time
long getTimeMillis()
{
	SYSTEMTIME st;
	GetSystemTime(&st);
	return st.wHour * 60 * 60 * 1000 + st.wMinute * 60 * 1000 + st.wSecond * 1000 + st.wMilliseconds;
}

char hex[176 * 2 + 1];
int increment = 0;
int callInc = 0;
char old[4] = { 0x00, 0x00, 0x00, 0x00 }; // Used for detecting block hashing info changes


bool different(char* one, char* two, int length)
{
	int i = 0;
	for (; i < length; i++)
	{
		if (one[i] != two[i]) return true;
	}
	return false;
}

int deviceToUse = 0;
void getHeaderForWork(uint8_t *header)
{
	if (callInc % 200 == 0)
	{
		callInc = 0;
		FILE *fr;


		char fileName[16] = "headeroutXX.txt";
		fileName[9]  = (deviceToUse / 10) + 48;
		fileName[10] = (deviceToUse % 10) + 48;

		fr = fopen(fileName, "rt");
		fgets(hex, 352, fr);
		hex[352] = '\0';

		if (different(old, hex, 4))
		{
			old[0] = hex[0];
			old[1] = hex[1];
			old[2] = hex[2];
			old[3] = hex[3];

			printf(" Real: %s\n", hex);
		}

		fclose(fr);
	}
	 
	callInc++;
	unsigned char* bufferHeader = hexToByteArray(hex);
	memcpy(header, bufferHeader, headerSize);
	free(bufferHeader);
	increment++;
	long time = std::time(0);

	header[168] = (time & 0x000000FF);
	header[169] = (time & 0x0000FF00) >> 8;
	header[170] = (time & 0x00FF0000) >> 16;
	header[171] = (time & 0xFF000000) >> 24;
}

void nonceGrindcuda(cudaStream_t cudastream, uint32_t threads, uint32_t *blockHeader, uint32_t *midState, uint32_t *nonceOut)
{
	cudaError_t e = cudaGetLastError();
	nonceGrindc << <128, 768, 2048, cudastream >> >(blockHeader, midState, nonceOut);
	e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(e));
	}
}

long start = getTimeMillis();
int totalNonces = 0;
int offset = 0;
void grindNonces(uint32_t items_per_iter, int cycles_per_iter)
{
	static bool init = false;
	static uint32_t *nonceOut = nullptr;
	static uint8_t *blockHeader = nullptr;
	static uint8_t *midState = nullptr;
	cudaError_t ret;

	if (!init)
	{
		ret = cudaMallocHost(&nonceOut, 4);
		if (ret != cudaSuccess)
		{
			printf("ERROR ALLOCATION\n");
		}
		ret = cudaMallocHost(&blockHeader, 64);
		if (ret != cudaSuccess)
		{
			printf("ERROR ALLOCATION\n");
		}
		ret = cudaMallocHost(&midState, 32);
		if (ret != cudaSuccess)
		{
			printf("ERROR ALLOCATION\n");
		}
		init = true;
	}

	long startTime = getTimeMillis();
	int i;

	getHeaderForWork(blockHeader);
	*nonceOut = 0;

	/* Calculate midstate */
	uint32_t block[64];

	uint32_t h0 = 0x6a09e667;
	uint32_t h1 = 0xbb67ae85;
	uint32_t h2 = 0x3c6ef372;
	uint32_t h3 = 0xa54ff53a;
	uint32_t h4 = 0x510e527f;
	uint32_t h5 = 0x9b05688c;
	uint32_t h6 = 0x1f83d9ab;
	uint32_t h7 = 0x5be0cd19;

	uint32_t a = h0;
	uint32_t b = h1;
	uint32_t c = h2;
	uint32_t d = h3;
	uint32_t e = h4;
	uint32_t f = h5;
	uint32_t g = h6;
	uint32_t h = h7;

	/* 16 * 32 = 512 bits, the size of a chunk in SHA-256 */
	for (int i = 0; i < 16; i++)
	{
		block[i] = ((uint32_t)blockHeader[i * 4 + 0] << 24) | ((uint32_t)blockHeader[i * 4 + 1] << 16) | ((uint32_t)blockHeader[i * 4 + 2] << 8) | ((uint32_t)blockHeader[i * 4 + 3]);
	}

	for (int i = 16; i < 64; i++)
	{
		block[i] = block[i - 16] + block[i - 7] + SIG1(block[i - 2]) + SIG0(block[i - 15]);
	}

	uint32_t temp1;
	uint32_t temp2;
	uint32_t S1;
	uint32_t S0;

	for (int i = 0; i < 64; i++)
	{
		S1 = (ROTRIGHT(e, 6)) ^ (ROTRIGHT(e, 11)) ^ (ROTRIGHT(e, 25));
		temp1 = h + S1 + ((e & f) ^ ((~e) & g)) + k[i] + block[i];
		S0 = (ROTRIGHT(a, 2)) ^ (ROTRIGHT(a, 13)) ^ (ROTRIGHT(a, 22));
		temp2 = S0 + (((a & b) ^ (a & c) ^ (b & c)));

		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}

	h0 += a;
	h1 += b;
	h2 += c;
	h3 += d;
	h4 += e;
	h5 += f;
	h6 += g;
	h7 += h;

	/* Now we do most if it again for the 2nd expansion/compression of the 2nd block (bits 513 to 1024): */

	a = h0;
	b = h1;
	c = h2;
	d = h3;
	e = h4;
	f = h5;
	g = h6;
	h = h7;

	/* 16 * 32 = 512 bits, the size of a chunk in SHA-256 */
	for (int i = 0; i < 16; i++)
	{
		block[i] = ((uint32_t)blockHeader[(i + 16) * 4 + 0] << 24) | ((uint32_t)blockHeader[(i + 16) * 4 + 1] << 16) | ((uint32_t)blockHeader[(i + 16) * 4 + 2] << 8) | ((uint32_t)blockHeader[(i + 16) * 4 + 3]);
	}

	for (int i = 16; i < 64; i++)
	{
		block[i] = block[i - 16] + block[i - 7] + SIG1(block[i - 2]) + SIG0(block[i - 15]);
	}

	for (int i = 0; i < 64; i++)
	{
		S1 = (ROTRIGHT(e, 6)) ^ (ROTRIGHT(e, 11)) ^ (ROTRIGHT(e, 25));
		temp1 = h + S1 + ((e & f) ^ ((~e) & g)) + k[i] + block[i];
		S0 = (ROTRIGHT(a, 2)) ^ (ROTRIGHT(a, 13)) ^ (ROTRIGHT(a, 22));
		temp2 = S0 + (((a & b) ^ (a & c) ^ (b & c)));

		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}

	h0 += a;
	h1 += b;
	h2 += c;
	h3 += d;
	h4 += e;
	h5 += f;
	h6 += g;
	h7 += h;

	uint32_t midstateInternal[8];

	midstateInternal[0] = h0;
	midstateInternal[1] = h1;
	midstateInternal[2] = h2;
	midstateInternal[3] = h3;
	midstateInternal[4] = h4;
	midstateInternal[5] = h5;
	midstateInternal[6] = h6;
	midstateInternal[7] = h7;

	uint32_t remainingHeader[17];
	for (int i = 0; i < 12; i++)
	{
		remainingHeader[i] = ((uint32_t)blockHeader[(i + 32) * 4 + 0] << 24) | ((uint32_t)blockHeader[(i + 32) * 4 + 1] << 16) | ((uint32_t)blockHeader[(i + 32) * 4 + 2] << 8) | ((uint32_t)blockHeader[(i + 32) * 4 + 3]);
	}
	remainingHeader[12] = 0x80000000;
	remainingHeader[13] = 0x00000000;
	remainingHeader[14] = 0x00000000;
	remainingHeader[15] = 0x00000580;

	for (i = 0; i < 1; i++)
	{
		remainingHeader[16] = ++offset;

		if (offset > 1024) {
			offset = 0;
		}

		ret = cudaSuccess;
		ret = cudaMemcpyAsync(blockHeadermobj, remainingHeader, 68, cudaMemcpyHostToDevice, cudastream);
		if (ret != cudaSuccess)
		{
			printf("Failed here1!\n");
		}

		ret = cudaSuccess;


		ret = cudaMemcpyAsync(midStatemobj, midstateInternal, 32, cudaMemcpyHostToDevice, cudastream);
		if (ret != cudaSuccess)
		{
			printf("Failed here2!\n");
		}

		ret = cudaMemcpyAsync(nonceOutmobj, nonceOut, 4, cudaMemcpyHostToDevice, cudastream);
		if (ret != cudaSuccess)
		{
			printf("Failed here3!\n");
		}
		nonceGrindcuda(cudastream, items_per_iter, blockHeadermobj, midStatemobj, nonceOutmobj);

		ret = cudaMemcpyAsync(nonceOut, nonceOutmobj, 4, cudaMemcpyDeviceToHost, cudastream);
		if (ret != cudaSuccess)
		{
			printf("Failed here!4\n");
		}
		ret = cudaStreamSynchronize(cudastream);
		if (ret != cudaSuccess)
		{
			printf("Failed here!5\n");
		}

		if (*nonceOut != 0)
		{
			uint32_t nonce = *nonceOut;
			nonce = (((nonce & 0xFF000000) >> 24) | ((nonce & 0x00FF0000) >> 8) | ((nonce & 0x0000FF00) << 8) | ((nonce & 0x000000FF) << 24));
			uint32_t timestamp = remainingHeader[10];
			timestamp = ((timestamp & 0x000000FF) << 24) + ((timestamp & 0x0000FF00) << 8) + ((timestamp & 0x00FF0000) >> 8) + ((timestamp & 0xFF000000) >> 24);
			printf("Found nonce: %08x    T: %08x    Hashrate: %.3f MH/s   Total: %d\n", nonce, timestamp, (((((double)totalNonces) * 4 * 16 * 16 * 16 * 16) / (4)) / (((double)getTimeMillis() - start) / 1000)), totalNonces);

			
			FILE* f2;

			char fileName[13] = "datainXX.txt";
			fileName[6] = (deviceToUse / 10) + 48;
			fileName[7] = (deviceToUse % 10) + 48;
			printf("Reading from %s\n", fileName);

			f2 = fopen(fileName, "w");
			while (f2 == NULL)
			{
				f2 = fopen(fileName, "w");
			}

			fprintf(f2, "\$%08x\n", nonce);
			fprintf(f2, "\$%08x", timestamp);
			fclose(f2); 
			

			*nonceOut = 0;
			totalNonces++;
		}
	}

	long endTime = getTimeMillis();
	double timeDelta = endTime - startTime;
}

int main(int argc, char *argv[])
{
	int i = 0; int j = 0;

	if (argc > 1)
	{
		for (i = 1; i < argc; i++)
		{
			char* argument = argv[i];
			if (argument[0] == 'd')
			{
				deviceToUse = argument[1] - 48;
			}
		}
	}

	printf("Using Device: %d\n\n", deviceToUse);

	unsigned int items_per_iter = 256 * 256 * 256 * 16;

	unsigned int cycles_per_iter = 15;
	double seconds_per_iter = 10.0;

	int version, ret;
	ret = cudaDriverGetVersion(&version);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}

	int deviceCount;
	ret = cudaGetDeviceCount(&deviceCount);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}

	cudaDeviceProp deviceProp;

	printf("CUDA Version: %.1f\n", ((float)version / 1000));
	printf("CUDA Devices: %d\n", deviceCount);

	printf("\n");

	for (int count = 0; count < deviceCount; count++)
	{
		ret = cudaGetDeviceProperties(&deviceProp, count);
		if (ret != cudaSuccess)
		{
			printf("ERROR ALLOCATION\n");
		}
		printf("Device #%d (%s):\n", count, deviceProp.name);
		printf("    Clock Rate:              %d MHz\n", (deviceProp.clockRate / 1024));
		printf("    Is Integrated:           %s\n", (deviceProp.integrated == 0 ? "false" : "true"));
		printf("    Compute Capability:      %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("    Kernel Concurrency:      %d\n", deviceProp.concurrentKernels);
		printf("    Max Grid Size:           %d\n", deviceProp.maxGridSize);
		printf("    Max Threads per Block:   %d\n", deviceProp.maxThreadsPerBlock);
		printf("    Registers per Block:     %d\n", deviceProp.regsPerBlock);
		printf("    Registers per SM:        %d\n", deviceProp.regsPerMultiprocessor);
		printf("    Processor Count:         %d\n", deviceProp.multiProcessorCount);
		printf("    Shared Memory/Block:     %d\n", deviceProp.sharedMemPerBlock);
		printf("    Shared Memory/Proc:      %d\n", deviceProp.sharedMemPerMultiprocessor);
		printf("    Warp Size:               %d\n", deviceProp.warpSize);
		printf("\n");
	}

	printf("Mining on device #%d...\n\n", deviceToUse);
	ret = cudaSetDevice(deviceToUse);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}
	cudaDeviceReset();
	ret = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}
	ret = cudaStreamCreate(&cudastream);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}
	ret = cudaMalloc(&blockHeadermobj, 68);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}
	ret = cudaMalloc(&midStatemobj, 32);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}
	ret = cudaMalloc(&nonceOutmobj, 4);
	if (ret != cudaSuccess)
	{
		printf("ERROR ALLOCATION\n");
	}


	cudaError_t e = cudaGetLastError();
	printf("Last error: %s\n", cudaGetErrorString(e));


	long start = getTimeMillis();
	grindNonces(items_per_iter, 1);

	float elapsedTime = getTimeMillis() - start;
	items_per_iter *= (seconds_per_iter / elapsedTime) / cycles_per_iter;

	bool quit = false;

	while (!quit)
	{
		grindNonces(items_per_iter, cycles_per_iter);
	}
}
