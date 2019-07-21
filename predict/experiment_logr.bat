time /t

python vpm_predict_tuned.py -f shin.csv -s 100 -t .7 -c logr -o shin_tuned_logr > tuned\shin_logr.txt

time /t

python vpm_predict_tuned.py -f zimmermann.csv -s 100 -t .7 -c logr -o zimmermann_tuned_logr > tuned\zimmermann_logr.txt

time /t

python vpm_predict_tuned.py -f scandariato.csv -s 100 -t .7 -c logr -o scandariato_tuned_logr  > tuned\scandariato_logr.txt

time /t

python vpm_predict_tuned.py -f everything.csv -s 100 -t .7 -c logr -o everything_tuned_logr > tuned\everything_logr.txt

time /t

python vpm_predict_tuned.py -f scandariato_crashes.csv -s 100 -t .7 -c logr -o scandariato_crashes_tuned_logr > tuned\scandariato_crashes_logr.txt

time /t

python vpm_predict_tuned.py -f textmining_softwaremetrics.csv -s 100 -t .7 -c logr -o textmining_softwaremetrics_tuned_logr > tuned\textmining_softwaremetrics_logr.txt

time /t

python vpm_predict_tuned.py -f softwaremetrics_crashes.csv -s 100 -t .7 -c logr -o softwaremetrics_crashes_tuned_logr > tuned\softwaremetrics_crashes_logr.txt
