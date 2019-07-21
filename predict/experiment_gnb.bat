time /t

python vpm_predict_tuned.py -f shin.csv -s 100 -t .7 -c gnb -o shin_tuned_gnb > tuned\shin_gnb.txt

time /t

python vpm_predict_tuned.py -f zimmermann.csv -s 100 -t .7 -c gnb -o zimmermann_tuned_gnb > tuned\zimmermann_gnb.txt

time /t

python vpm_predict_tuned.py -f scandariato.csv -s 100 -t .7 -c gnb -o scandariato_tuned_gnb  > tuned\scandariato_gnb.txt

time /t

python vpm_predict_tuned.py -f everything.csv -s 100 -t .7 -c gnb -o everything_tuned_gnb > tuned\everything_gnb.txt

time /t

python vpm_predict_tuned.py -f scandariato_crashes.csv -s 100 -t .7 -c gnb -o scandariato_crashes_tuned_gnb > tuned\scandariato_crashes_gnb.txt

time /t

python vpm_predict_tuned.py -f textmining_softwaremetrics.csv -s 100 -t .7 -c gnb -o textmining_softwaremetrics_tuned_gnb > tuned\textmining_softwaremetrics_gnb.txt

time /t

python vpm_predict_tuned.py -f softwaremetrics_crashes.csv -s 100 -t .7 -c gnb -o softwaremetrics_crashes_tuned_gnb > tuned\softwaremetrics_crashes_gnb.txt
