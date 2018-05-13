/*
 Proxy log anonymiser -- by David Alexander (with SHA1 code from Paul E. Jones)
 */

// Headers for SHA1 algorithm (see below)

typedef struct SHA1Context {
	unsigned Message_Digest[5];
		
	unsigned Length_Low;
	unsigned Length_High;
		
	unsigned char Message_Block[64];
	int Message_Block_Index;
		
	int Computed;
	int Corrupted;
} SHA1Context;

void SHA1Reset(SHA1Context*);
int SHA1Result(SHA1Context*);
void SHA1Input(SHA1Context*, const unsigned char*, unsigned);

// End headers for SHA1 algorithm.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define MAX_STR_LEN 1024

int main(int argc, char *argv[]) {
	SHA1Context sha;
	
	char user[MAX_STR_LEN+1], timestamp[MAX_STR_LEN+1], method[MAX_STR_LEN+1],
	     url[MAX_STR_LEN+1], protocol[MAX_STR_LEN+1];
	
	int status, bytes;
	int param;
        int ch;

        // Generate a long random string to concatenate to each username before hashing it.
        // This means that we can't find which log entries belong to a known username.
        srand(time(NULL));
        char rand_str[MAX_STR_LEN+1];
        for (ch = 0; ch<MAX_STR_LEN; ch++) {
           rand_str[ch] = 'a' + (rand()%26);
        }
        rand_str[MAX_STR_LEN] = '\0';

	for (param = 1; param < argc; param++) {
		
		FILE *infile = fopen(argv[param], "r");
		
		if (infile == NULL) {
			fprintf(stderr, "%s: fopen(\"%s\", \"r\"): ", argv[0], argv[param]);
			perror(NULL);
			continue;
		}
	
		FILE *outfile = stdout;
		
		fprintf(stderr, "Processing %s...\n", argv[param]);
		
		/*
		 Input format:
		   <IP address> - <username> [<timestamp (DD/Mon/YYYY:HH:MM:SS)> <timezone>]
		   "<HTTP method> <URL> <HTTP version>" <HTTP status> <bytes received> <TCP status>
		 Output format:
		   <hashed username> <HTTP method> <URL> <HTTP status> <bytes received> <timestamp (DD/Mon/YYYY:HH:MM:SS)> <original log filename>
		 */
		while (fscanf(infile, "%*s - %1024s [%1024s %*s \"%1024s %1024s %*s %d %d %*s\n",
					  user, timestamp, method, url, &status, &bytes) != EOF) {
			
			// If the log gives the username as "-", this usually means the request failed, so we'll ignore it.
			// Also, we're only interested in requests for Wikipedia pages.
			if (strcmp(user, "-") != 0 && strcasestr(url, "wikipedia") != NULL) {
			
				// Calculate and print the SHA1 hash of the username.
				SHA1Reset(&sha);
				SHA1Input(&sha, (unsigned char*)user, strlen(user));
                                SHA1Input(&sha, (unsigned char*)rand_str, MAX_STR_LEN);

				if (!SHA1Result(&sha)) {
					fprintf(outfile, "sha_error");
				}else{
					fprintf(outfile, "%08x%08x%08x%08x%08x",
							sha.Message_Digest[0], sha.Message_Digest[1],
							sha.Message_Digest[2], sha.Message_Digest[3],
							sha.Message_Digest[4]);
				}
				
				fprintf(outfile, " %s %s %d %d %s %s\n", method, url, status, bytes, timestamp, argv[param]);
				
			}
		}
		
	}
	
	fprintf(stderr, "Done.\n");
	return 0;
}


/*
 The following code implements the SHA1 hashing alogrithm, and is taken from http://www.packetizer.com/security/sha1/
 */

#define SHA1CircularShift(bits,word) \
((((word) << (bits)) & 0xFFFFFFFF) | \
((word) >> (32-(bits))))

void SHA1ProcessMessageBlock(SHA1Context *);
void SHA1PadMessage(SHA1Context *);

void SHA1Reset(SHA1Context *context)
{
    context->Length_Low             = 0;
    context->Length_High            = 0;
    context->Message_Block_Index    = 0;
	
    context->Message_Digest[0]      = 0x67452301;
    context->Message_Digest[1]      = 0xEFCDAB89;
    context->Message_Digest[2]      = 0x98BADCFE;
    context->Message_Digest[3]      = 0x10325476;
    context->Message_Digest[4]      = 0xC3D2E1F0;
	
    context->Computed   = 0;
    context->Corrupted  = 0;
}

int SHA1Result(SHA1Context *context)
{
	
    if (context->Corrupted)
    {
        return 0;
    }
	
    if (!context->Computed)
    {
        SHA1PadMessage(context);
        context->Computed = 1;
    }
	
    return 1;
}

void SHA1Input(     SHA1Context         *context,
			   const unsigned char *message_array,
			   unsigned            length)
{
    if (!length)
    {
        return;
    }
	
    if (context->Computed || context->Corrupted)
    {
        context->Corrupted = 1;
        return;
    }
	
    while(length-- && !context->Corrupted)
    {
        context->Message_Block[context->Message_Block_Index++] =
		(*message_array & 0xFF);
		
        context->Length_Low += 8;
        /* Force it to 32 bits */
        context->Length_Low &= 0xFFFFFFFF;
        if (context->Length_Low == 0)
        {
            context->Length_High++;
            /* Force it to 32 bits */
            context->Length_High &= 0xFFFFFFFF;
            if (context->Length_High == 0)
            {
                /* Message is too long */
                context->Corrupted = 1;
            }
        }
		
        if (context->Message_Block_Index == 64)
        {
            SHA1ProcessMessageBlock(context);
        }
		
        message_array++;
    }
}

void SHA1ProcessMessageBlock(SHA1Context *context)
{
    const unsigned K[] =            /* Constants defined in SHA-1   */      
    {
        0x5A827999,
        0x6ED9EBA1,
        0x8F1BBCDC,
        0xCA62C1D6
    };
    int         t;                  /* Loop counter                 */
    unsigned    temp;               /* Temporary word value         */
    unsigned    W[80];              /* Word sequence                */
    unsigned    A, B, C, D, E;      /* Word buffers                 */
	
    /*
     *  Initialize the first 16 words in the array W
     */
    for(t = 0; t < 16; t++)
    {
        W[t] = ((unsigned) context->Message_Block[t * 4]) << 24;
        W[t] |= ((unsigned) context->Message_Block[t * 4 + 1]) << 16;
        W[t] |= ((unsigned) context->Message_Block[t * 4 + 2]) << 8;
        W[t] |= ((unsigned) context->Message_Block[t * 4 + 3]);
    }
	
    for(t = 16; t < 80; t++)
    {
		W[t] = SHA1CircularShift(1,W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]);
    }
	
    A = context->Message_Digest[0];
    B = context->Message_Digest[1];
    C = context->Message_Digest[2];
    D = context->Message_Digest[3];
    E = context->Message_Digest[4];
	
    for(t = 0; t < 20; t++)
    {
        temp =  SHA1CircularShift(5,A) +
		((B & C) | ((~B) & D)) + E + W[t] + K[0];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }
	
    for(t = 20; t < 40; t++)
    {
        temp = SHA1CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[1];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }
	
    for(t = 40; t < 60; t++)
    {
        temp = SHA1CircularShift(5,A) +
		((B & C) | (B & D) | (C & D)) + E + W[t] + K[2];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }
	
    for(t = 60; t < 80; t++)
    {
        temp = SHA1CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[3];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }
	
    context->Message_Digest[0] =
	(context->Message_Digest[0] + A) & 0xFFFFFFFF;
    context->Message_Digest[1] =
	(context->Message_Digest[1] + B) & 0xFFFFFFFF;
    context->Message_Digest[2] =
	(context->Message_Digest[2] + C) & 0xFFFFFFFF;
    context->Message_Digest[3] =
	(context->Message_Digest[3] + D) & 0xFFFFFFFF;
    context->Message_Digest[4] =
	(context->Message_Digest[4] + E) & 0xFFFFFFFF;
	
    context->Message_Block_Index = 0;
}

void SHA1PadMessage(SHA1Context *context)
{
    if (context->Message_Block_Index > 55)
    {
        context->Message_Block[context->Message_Block_Index++] = 0x80;
        while(context->Message_Block_Index < 64)
        {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }
		
        SHA1ProcessMessageBlock(context);
		
        while(context->Message_Block_Index < 56)
        {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }
    }
    else
    {
        context->Message_Block[context->Message_Block_Index++] = 0x80;
        while(context->Message_Block_Index < 56)
        {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }
    }
	
    context->Message_Block[56] = (context->Length_High >> 24) & 0xFF;
    context->Message_Block[57] = (context->Length_High >> 16) & 0xFF;
    context->Message_Block[58] = (context->Length_High >> 8) & 0xFF;
    context->Message_Block[59] = (context->Length_High) & 0xFF;
    context->Message_Block[60] = (context->Length_Low >> 24) & 0xFF;
    context->Message_Block[61] = (context->Length_Low >> 16) & 0xFF;
    context->Message_Block[62] = (context->Length_Low >> 8) & 0xFF;
    context->Message_Block[63] = (context->Length_Low) & 0xFF;
	
    SHA1ProcessMessageBlock(context);
}
