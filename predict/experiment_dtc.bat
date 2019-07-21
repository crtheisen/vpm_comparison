time /t

python vpm_predict_tuned.py -f shin.csv -s 100 -t .7 -c dtc -o shin_tuned_dtc > tuned\shin_dtc.txt

time /t

python vpm_predict_tuned.py -f zimmermann.csv -s 100 -t .7 -c dtc -o zimmermann_tuned_dtc > tuned\zimmermann_dtc.txt

time /t

python vpm_predict_tuned.py -f scandariato.csv -s 100 -t .7 -c dtc -o scandariato_tuned_dtc  > tuned\scandariato_dtc.txt

time /t

python vpm_predict_tuned.py -f everything.csv -s 100 -t .7 -c dtc -o everything_tuned_dtc > tuned\everything_dtc.txt

time /t

python vpm_predict_tuned.py -f scandariato_crashes.csv -s 100 -t .7 -c dtc -o scandariato_crashes_tuned_dtc > tuned\scandariato_crashes_dtc.txt

time /t

python vpm_predict_tuned.py -f textmining_softwaremetrics.csv -s 100 -t .7 -c dtc -o textmining_softwaremetrics_tuned_dtc > tuned\textmining_softwaremetrics_dtc.txt

time /t

python vpm_predict_tuned.py -f softwaremetrics_crashes.csv -s 100 -t .7 -c dtc -o softwaremetrics_crashes_tuned_dtc > tuned\softwaremetrics_crashes_dtc.txt
