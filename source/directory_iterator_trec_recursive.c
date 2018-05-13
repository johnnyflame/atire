/*
	 DIRECTORY_ITERATOR_TREC_RECURSIVE.C
	 -----------------------------------
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "memory.h"
#include "instream_file.h"
#include "instream_deflate.h"
#include "instream_buffer.h"
#include "instream_scrub.h"
#include "directory_iterator_scrub.h"
#include "directory_iterator_recursive.h"
#include "directory_iterator_file_buffered.h"
#include "directory_iterator_trec_recursive.h"

/*
	ANT_DIRECTORY_ITERATOR_TREC_RECURSIVE::ANT_DIRECTORY_ITERATOR_TREC_RECURSIVE()
	------------------------------------------------------------------------------
*/
ANT_directory_iterator_trec_recursive::ANT_directory_iterator_trec_recursive(char *source, long get_file, long long scrubbing_options) : ANT_directory_iterator("", get_file)
{
ANT_directory_iterator_object filename;

this->source = source;
filename_provider = new ANT_directory_iterator_recursive(source, 0);

this->scrubbing_options = scrubbing_options;

more_files = filename_provider->first(&filename);
first_time = true;

file_stream = NULL;
decompressor = NULL;
instream_buffer = NULL;
scrubber = NULL;
detrecer = NULL;
memory = NULL;

new_provider(filename.filename);
}

/*
	ANT_DIRECTORY_ITERATOR_TREC_RECURSIVE::~ANT_DIRECTORY_ITERATOR_TREC_RECURSIVE()
	-------------------------------------------------------------------------------
*/
ANT_directory_iterator_trec_recursive::~ANT_directory_iterator_trec_recursive()
{
delete detrecer;
delete memory;
}

/*
	ANT_DIRECTORY_ITERATOR_TREC_RECURSIVE::NEW_PROVIDER()
	-----------------------------------------------------
*/
ANT_directory_iterator *ANT_directory_iterator_trec_recursive::new_provider(char *filename)
{
delete detrecer;
delete memory;

memory = new ANT_memory(1024 * 1024);
file_stream = new ANT_instream_file(memory, filename);

if (strcmp(filename + strlen(filename) - 3, ".gz") == 0)
	decompressor = new ANT_instream_deflate(memory, file_stream);
else
	decompressor = file_stream;

/*
	We only want a buffer in this position if it comes after a decompressor
*/
if (decompressor == instream_buffer)
	instream_buffer = decompressor;
else
	instream_buffer = new ANT_instream_buffer(memory, decompressor, false);

if (scrubbing_options != ANT_directory_iterator_scrub::NONE)
	scrubber = new ANT_instream_scrub(memory, instream_buffer, scrubbing_options);
else
	scrubber = instream_buffer;

detrecer = new ANT_directory_iterator_file_buffered(scrubber, ANT_directory_iterator::READ_FILE);

return detrecer;
}

/*
	ANT_DIRECTORY_ITERATOR_TREC_RECURSIVE::NEXT()
	---------------------------------------------
*/
ANT_directory_iterator_object *ANT_directory_iterator_trec_recursive::next(ANT_directory_iterator_object *object)
{
ANT_directory_iterator_object *got;

while (more_files != NULL)
	{
	if (first_time)
		got = detrecer->first(object);
	else
		got = detrecer->next(object);

	first_time = false;

	if (got == NULL)
		{
		if ((more_files = filename_provider->next(object)) != NULL)
			{
			new_provider(object->filename);
			first_time = true;
			}
		}
	else
		return got;
	}

return NULL;
}
