python3 train_embed_based.py --do_train \
			     --dataset="test" \
			     --model_type="textlstm" \
			     --hidden_size=240 \
			     --max_seq_length=64 \
			     --train_batch_size=4 \
			     --num_train_epochs=20 \
			     --learning_rate=5e-2 \
			     --dropout=0.5
