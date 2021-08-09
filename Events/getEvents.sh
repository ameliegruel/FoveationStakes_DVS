debut=`date +"%s"`
python high2lowResolution.py events_dataset_test_HR.npy
fin=`date +"%s"`
echo "Temps de calcul :" $(($fin-$debut))