bert_preprocessing_command = 'python my_pretraining.py'
bert_preprocessing_command += ' --input_dir=' + 'D:\database\wiketext\sub_new\en_30528_hdf5'
bert_preprocessing_command += ' --config_file=' + 'base.json'
bert_preprocessing_command += ' --output_dir=' + 'output'
bert_preprocessing_command += ' --train_batch_size='+str(12)
bert_preprocessing_command += ' --learning_rate=' + str(5e-5)
bert_preprocessing_command += ' --num_train_epochs=' + str(10)
bert_preprocessing_command += ' --max_steps=' + str(1000)
bert_preprocessing_command += ' --warmup_proportion=' + str(0.01)
bert_preprocessing_command += ' --resume_from_checkpoint=' + str(True)
# bert_preprocessing_command += ' --fp16=' + str(True)
bert_preprocessing_command += ' --num_workers=' + str(4)
print(bert_preprocessing_command)