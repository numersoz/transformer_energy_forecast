# informer

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 48 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 24 --learning_rate 0.0001 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 96 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 24 --learning_rate 0.0001 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 144 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 72 --learning_rate 0.0001 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 168 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 72 --learning_rate 0.0001 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 240 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 168 --learning_rate 0.0001 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 336 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0001 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 400 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0001 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 48 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 24 --learning_rate 0.0003 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 96 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 24 --learning_rate 0.0003 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 144 --label_len 96 --enc_in 6 --dec_in 6 --pred_len 72 --learning_rate 0.0003 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 168 --label_len 96 --enc_in 6 --dec_in 6 --pred_len 72 --learning_rate 0.0003 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 240 --label_len 120 --enc_in 6 --dec_in 6 --pred_len 168 --learning_rate 0.0003 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 336 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 400 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 400 --learning_rate 0.0003 --attn prob --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 48 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 48 --learning_rate 0.0001 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 96 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 48 --learning_rate 0.0001 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 144 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 48 --learning_rate 0.0001 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 168 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 48 --learning_rate 0.0001 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 240 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 48 --learning_rate 0.0001 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 336 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 48 --learning_rate 0.0001 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 400 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 48 --learning_rate 0.0001 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 48 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 96 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 144 --label_len 96 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 168 --label_len 96 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 240 --label_len 120 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 336 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn full --dropout 0.20 --lradj type4

python -u main_informer.py --model informer --data custom --root_path=./data/ --data_path elec_data.csv --features MS --target electricity_demand_values --freq h --batch_size 32 --train_epochs 10 --des "elec" --loss mse --itr 5 --seq_len 400 --label_len 48 --enc_in 6 --dec_in 6 --pred_len 336 --learning_rate 0.0003 --attn full --dropout 0.20 --lradj type4

# lstm

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 48 --label_len 48 --enc_in 12 --dec_in 6 --pred_len 24 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 96 --label_len 48 --enc_in 12 --dec_in 6 --pred_len 24 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 168 --label_len 96 --enc_in 12 --dec_in 6 --pred_len 24 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 240 --label_len 96 --enc_in 12 --dec_in 6 --pred_len 24 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 48 --label_len 48 --enc_in 12 --dec_in 6 --pred_len 72 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 96 --label_len 48 --enc_in 12 --dec_in 6 --pred_len 72 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 168 --label_len 96 --enc_in 12 --dec_in 6 --pred_len 72 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 240 --label_len 96 --enc_in 12 --dec_in 6 --pred_len 72 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 48 --label_len 48 --enc_in 12 --dec_in 6 --pred_len 168 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 96 --label_len 48 --enc_in 12 --dec_in 6 --pred_len 168 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 144 --label_len 72 --enc_in 12 --dec_in 6 --pred_len 168 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 168 --label_len 96 --enc_in 12 --dec_in 6 --pred_len 168 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

python -u main_informer.py --model dto-drnn --data custom --root_path=./data/ --data_path date_elec.csv --features MS --target electricity_demand_values --freq h --batch_size 64 --train_epochs 40 --des "dto" --loss mse --itr 5 --seq_len 240 --label_len 96 --enc_in 12 --dec_in 6 --pred_len 168 --learning_rate 0.003 --attn prob --dropout 0.10 --lradj type5 --d_model 300 --clip 10 --weight 0.0001 

# and so on...