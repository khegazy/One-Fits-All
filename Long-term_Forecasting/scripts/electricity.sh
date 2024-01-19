export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS

for pred_len in 96 #192 336 720
do
for percent in 100
do

python3 main.py "$@"\
    --root_path /pscratch/sd/k/khegazy/datasets/time_series/electricity/consumer_load/ \
    --data_path electricity.csv \
    --model_id 'ECL_'$model \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 1024 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1
done
done
