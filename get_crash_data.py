import subprocess, csv

file_list = {}
crash_file_list = {}

with open('/Users/christophertheisen/research/vpm_comparison/all_crash_concat.csv', 'rU') as csvdata:
  reader = csv.DictReader(csvdata)
  for row in reader:
    crash_file_list[row['File']] = row['Count']


with open('/Users/christophertheisen/research/vpm_comparison/clean_firefox_metrics_security.csv', 'rU') as csvdata:
  reader = csv.DictReader(csvdata)
  for row in reader:
    if row['File'] in crash_file_list:
      file_list[row['File']] = crash_file_list[row['File']]
    else:
      file_list[row['File']] = 0



with open('clean_crash_metrics.csv', 'wb') as csv_file:
  writer = csv.writer(csv_file)
  for key, value in file_list.items():
    writer.writerow([key, value])
