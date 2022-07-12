bert_preprocessing_command = 'python run_squad.py'
bert_preprocessing_command += ' --bert_model=' + 'bert_pretrain'
bert_preprocessing_command += ' --init_checkpoint=' + 'check_point\pytorch_model.bin'
bert_preprocessing_command += ' --output_dir=' + 'out_dir'
bert_preprocessing_command += ' --train_file='+'squad\\v1.1\\train-v1.1.json'
bert_preprocessing_command += ' --predict_file=' +'squad\\v1.1\\dev-v1.1.json'
bert_preprocessing_command += ' --max_seq_length=' + str(384)
bert_preprocessing_command += ' --doc_stride=' + str(128)
bert_preprocessing_command += ' --warmup_proportion=' + str(0.01)
bert_preprocessing_command += ' --max_query_length=' + str(64)

bert_preprocessing_command += ' --train_batch_size=' + str(16)
bert_preprocessing_command += ' --predict_batch_size=' + str(4)
bert_preprocessing_command += ' --learning_rate=' + str(5e-5)
bert_preprocessing_command += ' --num_train_epochs=' + str(130)
bert_preprocessing_command += ' --cache_dir=' + 'squad\cached'

bert_preprocessing_command += ' --vocab_file=' + 'bert_pretrain\\vocab.txt'
bert_preprocessing_command += ' --config_file=' + 'bert_pretrain\\bert_config.json'

bert_preprocessing_command += ' --do_train=' + str(True)
bert_preprocessing_command += ' --do_predict=' + str(True)

print(bert_preprocessing_command)