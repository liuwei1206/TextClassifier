python3 train_embed_based.py --do_train \
			     --dataset="toefl_p1234" \
			     --model_type="textlstm" \
			     --text_key="relations" \
			     --hidden_size=240 \
			     --max_seq_length=64 \
			     --train_batch_size=32 \
			     --num_train_epochs=20 \
			     --learning_rate=1e-3 \
			     --dropout=0.2
