python3 train_embed_based.py --do_train \
			     --dataset="toefl_p1234_text" \
			     --model_type="textlstm" \
			     --text_key="text" \
			     --hidden_size=240 \
			     --max_seq_length=512 \
			     --train_batch_size=64 \
			     --num_train_epochs=20 \
			     --learning_rate=1e-3 \
			     --dropout=0.2
