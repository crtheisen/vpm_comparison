import requests, json, os, time, csv

crash_dict = {}

dir_file = open('list_to_concat.csv', "r")

i = 0

for crash_dir in dir_file:

	crash_dir = crash_dir.strip()
	crash_files = os.listdir('/Users/christophertheisen/research/vpm_comparison/' + crash_dir)

	for file in crash_files:
		
		try:
			reader = open(crash_dir + file, "r")
			print 'Parsing file: ' + file

			for line in reader:

				crash_file = line.split(':')

				try:
					if crash_file[2] in crash_dict:
						crash_dict[crash_file[2]] += 1
					else:
						crash_dict[crash_file[2]] = 1
				except:
					print 'invalid line, moving on...'

			reader.close()
		except:
			print 'Invalid file name: ' + file + ' - Continuing...'

	with open('crashes/' + str(i) + '_concat.csv', 'wb') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in crash_dict.items():
			writer.writerow([key, value])
	i += 1