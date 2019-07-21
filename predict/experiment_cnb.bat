time /t

python vpm_predict_tuned.py -f shin.csv -s 100 -t .7 -c cnb -o shin_tuned_cnb > tuned\shin_cnb.txt

time /t

python vpm_predict_tuned.py -f zimmermann.csv -s 100 -t .7 -c cnb -o zimmermann_tuned_cnb > tuned\zimmermann_cnb.txt

time /t

python vpm_predict_tuned.py -f scandariato.csv -s 100 -t .7 -c cnb -o scandariato_tuned_cnb  > tuned\scandariato_cnb.txt

time /t

python vpm_predict_tuned.py -f everything.csv -s 100 -t .7 -c cnb -o everything_tuned_cnb > tuned\everything_cnb.txt

time /t

python vpm_predict_tuned.py -f scandariato_crashes.csv -s 100 -t .7 -c cnb -o scandariato_crashes_tuned_cnb > tuned\scandariato_crashes_cnb.txt

time /t

python vpm_predict_tuned.py -f textmining_softwaremetrics.csv -s 100 -t .7 -c cnb -o textmining_softwaremetrics_tuned_cnb > tuned\textmining_softwaremetrics_cnb.txt

time /t

python vpm_predict_tuned.py -f softwaremetrics_crashes.csv -s 100 -t .7 -c cnb -o softwaremetrics_crashes_tuned_cnb > tuned\softwaremetrics_crashes_cnb.txt
