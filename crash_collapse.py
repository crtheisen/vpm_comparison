import requests, json, os, time, csv

crash_dict = {}

crash_files = os.listdir('/Users/christophertheisen/research/vpm_comparison/crashes/')

for file in crash_files:
	
	try:
		reader = open('/Users/christophertheisen/research/vpm_comparison/crashes/' + file, "r")
		print 'Parsing file: ' + file

		for line in reader:

			crash_file = line.split(',')

			try:
				if crash_file[0] in crash_dict:
					crash_dict[crash_file[0]] += int(crash_file[1])
				else:
					crash_dict[crash_file[0]] = int(crash_file[1])
			except:
				print 'invalid line, moving on...'

		reader.close()
	except:
		print 'Invalid file name: ' + file + ' - Continuing...'

with open('all_crash_concat.csv', 'wb') as csv_file:
	writer = csv.writer(csv_file)
	for key, value in crash_dict.items():
		writer.writerow([key, value])