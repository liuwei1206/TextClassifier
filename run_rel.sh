python3 train_embed_based.py --do_train \
			     --dataset="toefl_p1234_rel" \
			     --model_type="textlstm" \
			     --text_key="relations" \
			     --hidden_size=240 \
			     --max_seq_length=48 \
			     --train_batch_size=64 \
			     --num_train_epochs=20 \
			     --learning_rate=5e-3 \
			     --dropout=0.2
