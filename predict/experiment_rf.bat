time /t

python vpm_predict_tuned.py -f shin.csv -s 100 -t .7 -c rf -o shin_tuned_rf > tuned\shin_rf.txt

time /t

python vpm_predict_tuned.py -f zimmermann.csv -s 100 -t .7 -c rf -o zimmermann_tuned_rf > tuned\zimmermann_rf.txt

time /t

python vpm_predict_tuned.py -f scandariato.csv -s 100 -t .7 -c rf -o scandariato_tuned_rf  > tuned\scandariato_rf.txt

time /t

python vpm_predict_tuned.py -f everything.csv -s 100 -t .7 -c rf -o everything_tuned_rf > tuned\everything_rf.txt

time /t

python vpm_predict_tuned.py -f scandariato_crashes.csv -s 100 -t .7 -c rf -o scandariato_crashes_tuned_rf > tuned\scandariato_crashes_rf.txt

time /t

python vpm_predict_tuned.py -f textmining_softwaremetrics.csv -s 100 -t .7 -c rf -o textmining_softwaremetrics_tuned_rf > tuned\textmining_softwaremetrics_rf.txt

time /t

python vpm_predict_tuned.py -f softwaremetrics_crashes.csv -s 100 -t .7 -c rf -o softwaremetrics_crashes_tuned_rf > tuned\softwaremetrics_crashes_rf.txt
