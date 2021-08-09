input_data=$1
frame_interval=100

# haute résolution
debut=`date +"%s"`
output_file="events_HR.npy"
python high2lowResolution.py $input_data -HR -o $output_file #--frame_interval $frame_interval
python getNewFormalism.py $output_file 
fin=`date +"%s"`
echo "Temps de calcul pour haute résolution :" $(($fin-$debut)) "s"


# basse résolution
debut=`date +"%s"`
output_file="events_LR.npy"
python high2lowResolution.py $input_data -LR -o $output_file #--frame_interval $frame_interval
python getNewFormalism.py $output_file
fin=`date +"%s"`
echo "Temps de calcul pour basse résolution :" $(($fin-$debut)) "s"