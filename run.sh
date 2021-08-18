#!/bin/bash
# shopt -s extglob
#number of filters
for set_no in 1 2 3 4 5 6 7 8
do
for n_filters in 5 10 20 30 50 70 100
do
python3 train_cnn.py --data-path data/training_data.csv --model-type shallow --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters $n_filters --n-epochs 150 --filter-size-1 2 --filter-size-2 3 --filter-size-3 4 --dropout-rate 0.6 --strides 2 --ksize 3 --learning-power 4 --experiment 1 --variable $n_filters
done
done

#filter region size
arr1=(2 4 7 14 3)
arr2=(3 5 8 15 3)
arr3=(4 6 9 16 3)
for set_no in 1 2 3 4 5 6 7 8
do
for (( i=0; i<${#arr1[@]}; i++ ));
do
python3 train_cnn.py --data-path data/training_data.csv --model-type shallow --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --filter-size-1 ${arr1[i]} --filter-size-2 ${arr2[i]} --filter-size-3 ${arr3[i]} --dropout-rate 0.6 --strides 2 --ksize 3 --learning-power 4 --experiment 2 --variable ${arr1[i]}${arr2[i]}${arr3[i]}
done
done

#dropout
for set_no in 1 2 3 4 5 6 7 8
do
for dp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
python3 train_cnn.py --data-path data/training_data.csv --model-type shallow --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --filter-size-1 2 --filter-size-2 3 --filter-size-3 4 --dropout-rate $dp --strides 2 --ksize 3 --learning-power 4 --experiment 3 --variable $dp
done
done

#strides
for set_no in 1 2 3 4 5 6 7 8
do
for strides in 2 3 4 5
do
python3 train_cnn.py --data-path data/training_data.csv --model-type shallow --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --filter-size-1 2 --filter-size-2 3 --filter-size-3 4 --dropout-rate 0.6 --strides $strides --ksize 3 --learning-power 4 --experiment 4 --variable $strides
done
done

#ksize
for set_no in 1 2 3 4 5 6 7 8
do
for ksize in 3 4 5 6
do
python3 train_cnn.py --data-path data/training_data.csv --model-type shallow --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --filter-size-1 2 --filter-size-2 3 --filter-size-3 4 --dropout-rate 0.6 --strides 2 --ksize $ksize --learning-power 4 --experiment 5 --variable $ksize
done
done

#learning power
for set_no in 1 2 3 4 5 6 7 8
do
for learning_power in 3 4 5 6
do
python3 train_cnn.py --data-path data/training_data.csv --model-type shallow --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --filter-size-1 2 --filter-size-2 3 --filter-size-3 4 --dropout-rate 0.6 --strides 2 --ksize 3 --learning-power $learning_power --experiment 6 --variable $learning_power
done
done

#deep word
for set_no in 1 2 3 4 5 6 7 8
do
python3 train_cnn.py --data-path data/training_data.csv --model-type deep --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --dropout-rate 0.6 --strides 2 --ksize 3 --n-channels 1 --learning-power 4 --experiment 7
done

#deep word + one hot pos
for set_no in 1 2 3 4 5 6 7 8
do
python3 train_cnn.py --data-path data/training_data.csv --model-type deep-onehot --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --dropout-rate 0.6 --strides 2 --ksize 3 --n-channels 2 --learning-power 4 --experiment 8
done

#deep word + pos emb
for set_no in 1 2 3 4 5 6 7 8
do
python3 train_cnn.py --data-path data/training_data.csv --model-type deep --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --dropout-rate 0.6 --strides 2 --ksize 3 --n-channels 2 --learning-power 4 --experiment 9
done

#deep word + pos emb initializer type
for set_no in 1 2 3 4 5 6 7 8
do
for initializer_type in random-normal he-normal xavier
do
python3 train_cnn.py --data-path data/training_data.csv --model-type deep --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --dropout-rate 0.6 --strides 2 --ksize 3 --n-channels 2 --learning-power 4 --initializer-type $initializer_type --experiment 10 --variable $initializer_type
done
done

#deep word + pos emb 300d
for set_no in 1 2 3 4 5 6 7 8
do
python3 train_cnn.py --data-path data/training_data.csv --model-type deep --embedding-path data/glove.6B.300d.txt --embedding-size 300 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --dropout-rate 0.6 --strides 2 --ksize 3 --n-channels 2 --learning-power 4 --initializer-type xavier --experiment 11
done

#train pos embeddings on all sets
python3 train_cnn.py --data-path data/training_data.csv --model-type deep --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --n-runs 1 --n-filters 5 --n-epochs 150 --dropout-rate 0.6 --strides 2 --ksize 3 --n-channels 2 --learning-power 4 --experiment 12 --train-all-sets --initializer-type xavier

#test trained pos embedding on all sets
for set_no in 1 2 3 4 5 6 7 8
do
python3 train_cnn.py --data-path data/training_data.csv --model-type deep --embedding-path data/glove.6B.50d.txt --embedding-size 50 --train-overall --set-no $set_no --n-runs 50 --n-filters 5 --n-epochs 150 --dropout-rate 0.6 --strides 2 --ksize 3 --n-channels 2 --learning-power 4 --experiment 13 --self-trained-pos-embedding --initializer-type xavier
done
