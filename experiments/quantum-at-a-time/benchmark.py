#
# Created : December-2012
#  Author : fei@cs.otago.ac.nz
#
# This script is used to collect profiling stats for quantum-at-a-time.
#
#
import sys
import os
import subprocess
import re
import csv
import time
from os.path import expanduser

class Profile:
	TOP_K_STR = 'Topk'
	QUERY_TIME_STR = 'Query Time'
	TOTAL_QUANTUM_STR = 'Total Quantum'
	PROCESSED_QUANTUM_STR = 'Processed Quantum'
	PROCESSED_POSTINGS_STR = 'Processed Postings'
	FINAL_MAP_STR = 'MAP'

	QUERY_TIME_PATTERN = re.compile(r'.*<time>([0-9]*)</time>', re.IGNORECASE)
	TOTAL_QUANTUM_PATTERN = re.compile(r'total quantums:\s*([0-9]*)', re.IGNORECASE)
	PROCESSED_QUANTUM_PATTERN = re.compile(r'processed quantums:\s*([0-9]*)', re.IGNORECASE)
	PROCESSED_POSTINGS_PATTERN = re.compile(r'processed postings:\s*([0-9]*)', re.IGNORECASE)
	FINAL_MAP_PATTERN = re.compile(r'\s*(MAP|nDCG).*:\s*([0-9.]*)', re.IGNORECASE)

	def __init__(self):
		self.top_k = 0;
		self.query_time = 0;
		self.total_quantum = 0;
		self.processed_quantum = 0;
		self.processed_postings = 0;
		self.final_map = 0.0;
	def __str__(self):
		output = "Topk: %d\nquery_time: %d\ntotal_quantum: %d\nprocessed_quantum: %d\nprocessed_postings: %d\nfinal_MAP: %f\n"
		return  output % (self.top_k, self.query_time, self.total_quantum, self.processed_quantum, self.processed_postings, self.final_map)
	def get_headers():
		return tuple((Profile.TOP_K_STR, Profile.QUERY_TIME_STR, Profile.TOTAL_QUANTUM_STR, Profile.PROCESSED_QUANTUM_STR, Profile.PROCESSED_POSTINGS_STR, Profile.FINAL_MAP_STR))
	get_headers = staticmethod(get_headers)
	def get_values(self):
		return tuple((self.top_k, self.query_time, self.total_quantum, self.processed_quantum, self.processed_postings, self.final_map))
pass # end of class profile


def run_benchmark(exe_file, prof_list, top_k, assessment_file, query_file, processing_stragety, precision_measure):
	args = list([exe_file])
	#args.extend(['-M'])
	#args.extend(['-Pq:d'])
	if precision_measure == 'MAP':
		args.append('-m%s' % precision_measure)
	elif precision_measure == 'MAP@' or precision_measure == 'nDCG@':
		args.append('-m%s%d' % (precision_measure, top_k))
	args.extend(['-%s' % processing_stragety])
	args.extend(['-k%d' % top_k])
	args.extend(['-a', assessment_file])
	args.extend(['-q', query_file])
	args.extend(['-l0'])

	print "the command is: %s\n" % args
	process = subprocess.Popen(args, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

	# catch the output of the search engine
	(child_stdout, child_stderr) = process.communicate()

	# check if there is an error before collecting the stats
	if (child_stderr != None):
		print child_stderr
		sys.exit(2)

	# create a new profile for the current run
	prof = Profile()
	prof.top_k = top_k
	#prof.query_time = 0.0
	#prof.total_quantum = 0
	#prof.processed_quantum = 0
	#prof.final_map = 0.0

	# split the output string and process line by line
	for l in child_stdout.split('\n'):
		result = Profile.QUERY_TIME_PATTERN.search(l)
		if result != None:
			prof.query_time += int(result.group(1))
		result = Profile.FINAL_MAP_PATTERN.search(l)
		if result != None:
			print 'the match is :%s' % result.group(2)
			prof.final_map= float(result.group(2))
		result = Profile.PROCESSED_QUANTUM_PATTERN.search(l)
		if result != None:
			prof.processed_quantum += int(result.group(1))
		result = Profile.PROCESSED_POSTINGS_PATTERN.search(l)
		if result != None:
			prof.processed_postings += int(result.group(1))
		result = Profile.TOTAL_QUANTUM_PATTERN.search(l)
		if result != None:
			prof.total_quantum += int(result.group(1))
	pass #end of for l in

	prof_list.append(prof)
	print prof
pass # end of def run_benchmark


def export_profile_data(prof_list, output_file="output.csv"):
	f = open(output_file, 'w')
	export = csv.writer(f, delimiter=',')
	export.writerow(Profile.get_headers())
	for p in prof_list:
		export.writerow(p.get_values())
	f.close()
pass # end of def export_profile_date


def usage(prog_name):
	print 'sage : %s [--exe the-executable] --ps processing_strategy [--outfile output-file] -a assessment-file -q query-file' % os.path.basename(prog_name)
	print '  --ps [M | Pq:n | Pq:d | Pd:s | Pq:l]'
	print
	sys.exit(2)
pass # end of def usgae(prog_name)


def test():
	print Profile.get_headers()
	sys.exit(2)
pass # end of def test()



def main():

	home_dir = expanduser('~')
	output_file = 'output.csv'
	exe_file = '%s/source-devel/hg/atire/bin/atire' % home_dir
	processing_stragety = None
	assessment_file = None
	query_file = None
	range = 'small' # either 'small' or 'large'
	#test()
	repeat = 5
	precision_measure = 'MAP' # or 'MAP@' or 'nDCG@'

	i = 1;
	while i < len(sys.argv):
		if sys.argv[i] == '-h' or sys.argv[i] == '--help':
			usage(sys.argv[0])
			i += 1
		elif sys.argv[i] == '--exe':
			i += 1
			exe_file = os.path.abspath(sys.argv[i])
			i += 1
		elif sys.argv[i] == '--ps':
			i += 1
			processing_stragety = sys.argv[i]
			i += 1
		elif sys.argv[i] == '--outfile':
			i += 1
			output_file = sys.argv[i]
			output_file = output_file.replace(':', '.')
			i += 1
		elif sys.argv[i] == '-a':
			i += 1
			assessment_file = sys.argv[i]
			i += 1
		elif sys.argv[i] == '-q':
			i += 1
			query_file = sys.argv[i]
			i += 1
		elif sys.argv[i] == '--range':
			i += 1
			range = sys.argv[i]
			i += 1
		elif sys.argv[i] == '--repeat':
			i += 1
			repeat = int(sys.argv[i])
			i += 1
		elif sys.argv[i] == '--precision-measure':
			i += 1
			precision_measure = sys.argv[i]
			i += 1
		else:
			print "un-recognised option: %s" % sys.argv[i]
			sys.exit(2)
		pass
	pass

	if assessment_file == '' or query_file == '':
		usage(sys.argv[0])
		sys.exit(2)

	top_k_list = []
	if range == 'small':
		top_k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
		#top_k_list = [10]
	else:
		top_k_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];

	#
	# all profiling data is stored in this variable
	#
	prof_list = list([])

	#
	# generate the parameters and call run_benchmark
	#
	for repeat in xrange(1,repeat+1):
		for top_k in top_k_list:
			print "repeat: %d, top_k: %d" % (repeat, top_k)
			run_benchmark(exe_file, prof_list, top_k, assessment_file, query_file, processing_stragety, precision_measure)
		pass
	pass


	#
	# save results to file
	#
	export_profile_data(prof_list, output_file)

pass #end of def main()


if __name__ == '__main__':
	main()
