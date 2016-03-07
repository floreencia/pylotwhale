#!/bin/bash

N=100
fileType=Freqs
echo "shufflings: $N"

for g in H #J B D F G
do
    echo "group $g"
    for i in {0..2}
    do
	echo "tau $i"

	#python ~/whales/scripts/NLP/csvDatFr-seqs-2gramN_shuffledDistributions.py ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds/NPWVR-seqsData.csv $g -T $i -N $N -n 0 -minC 10

	wait

	echo "\nplot bigram matrices"
	python ~/whales/scripts/NLP/csvShuffledDistributions-plotBigrams.py ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds-shuffled/shuffled${fileType}Dist_NPWVR-seqsData_DATE_TAPE_NSH${N}_GR${g}_TAU${i}.csv ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds-shuffled/dictionary_shuffled${fileType}Dist_NPWVR-seqsData_DATE_TAPE_NSH${N}_GR${g}_TAU${i}.dat #-rnCalls

	echo "chi-square test"

	#python ~/whales/scripts/NLP/csvShuffledDistributions-chiSqr.py ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds-shuffled/shuffled${fileType}Dist_NPWVR-seqsData__N${N}_${g}_T${i}.csv ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds-shuffled/dictionary_shuffled${fileType}Dist_NPWVR-seqsData__N${N}_${g}_T${i}.dat -af ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds-shuffled/confidenceIntervalsChi/summary.txt

	echo "diff proportions test"

	python ~/whales/scripts/NLP/csvShuffledDistributions-diffProportions.py ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds-shuffled/shuffled${fileType}Dist_NPWVR-seqsData__N${N}_${g}_T${i}.csv ~/whales/NLP/NPWvocalRepertoire/wPandas/no_general_sounds-shuffled/dictionary_shuffled${fileType}Dist_NPWVR-seqsData__N${N}_${g}_T${i}.dat -no-rnCalls 


    done
done


echo "done $N"
