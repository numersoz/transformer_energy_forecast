
# Transformer

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 100 --iterations 10 --seq_len 64 --dec_seq_len 48 --pred_len 24 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 4 --n_encoder_layers 4 --n_decoder_layers 4 --embedding_size 64 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 100 --iterations 10 --seq_len 64 --dec_seq_len 48 --pred_len 24 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 4 --n_encoder_layers 3 --n_decoder_layers 3 --embedding_size 64 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 100 --iterations 10 --seq_len 64 --dec_seq_len 48 --pred_len 24 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 3 --n_encoder_layers 3 --n_decoder_layers 3 --embedding_size 64 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 100 --iterations 10 --seq_len 64 --dec_seq_len 48 --pred_len 72 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 3 --n_encoder_layers 3 --n_decoder_layers 3 --embedding_size 128 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 100 --iterations 10 --seq_len 128 --dec_seq_len 48 --pred_len 72 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 3 --n_encoder_layers 3 --n_decoder_layers 3 --embedding_size 128 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 100 --iterations 10 --seq_len 128 --dec_seq_len 48 --pred_len 72 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 3 --n_encoder_layers 2 --n_decoder_layers 2 --embedding_size 128 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 128 --iterations 10 --seq_len 128 --dec_seq_len 48 --pred_len 168 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 2 --n_encoder_layers 2 --n_decoder_layers 2 --embedding_size 128 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 128 --iterations 10 --seq_len 128 --dec_seq_len 48 --pred_len 168 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 3 --n_encoder_layers 3 --n_decoder_layers 3 --embedding_size 128 --dropout 0.01 --lr 0.001

python train.py  --data ELECT1 --features M --target electricity --freq h --batch_size 128 --iterations 10 --seq_len 128 --dec_seq_len 48 --pred_len 168 --input_len 7 --output_len 7 --hidden_size 312 --n_heads 4 --n_encoder_layers 3 --n_decoder_layers 3 --embedding_size 196 --dropout 0.01 --lr 0.001

