#!/bin/bash

now=$(date +"%T")
echo "Running Shin"
echo "Current time : $now"
/usr/bin/python /Users/christophertheisen/research/vpm_comparison/predict/vpm_predict.py -f shin.csv -s 100 -t .7 -c gnb > shin.txt
echo "Running Zimmermann"
now=$(date +"%T")
echo "Current time : $now"
/usr/bin/python /Users/christophertheisen/research/vpm_comparison/predict/vpm_predict.py -f zimmermann.csv -s 100 -t .7 -c gnb > zimmermann.txt
echo "Running scandariato"
now=$(date +"%T")
echo "Current time : $now"
/usr/bin/python /Users/christophertheisen/research/vpm_comparison/predict/vpm_predict.py -f scandariato.csv -s 100 -t .7 -c gnb > scandariato.txt
echo "Running everything"
now=$(date +"%T")
echo "Current time : $now"
/usr/bin/python /Users/christophertheisen/research/vpm_comparison/predict/vpm_predict.py -f everything.csv -s 100 -t .7 -c gnb > everything.txt
echo "Running scandariato_crashes"
now=$(date +"%T")
echo "Current time : $now"
/usr/bin/python /Users/christophertheisen/research/vpm_comparison/predict/vpm_predict.py -f scandariato_crashes.csv -s 100 -t .7 -c gnb > scandariato_crashes.txt
echo "Running textmining_softwaremetrics"
now=$(date +"%T")
echo "Current time : $now"
/usr/bin/python /Users/christophertheisen/research/vpm_comparison/predict/vpm_predict.py -f textmining_softwaremetrics.csv -s 100 -t .7 -c gnb > textmining_softwaremetrics.txt
echo "Running softwaremetrics_crashes"
now=$(date +"%T")
echo "Current time : $now"
/usr/bin/python /Users/christophertheisen/research/vpm_comparison/predict/vpm_predict.py -f softwaremetrics_crashes.csv -s 100 -t .7 -c gnb > softwaremetrics_crashes.txt
now=$(date +"%T")
echo "Current time : $now"
