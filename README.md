# Pretraining-Train-Bert-From-Scratch
### change it to neat code
### Code From nvidia:https://github.com/NVIDIA/DeepLearningExamples
## How to run the code 
### first step:
preprocessing your text to specific training data that contain input_ids,segement ids,mask_ids,mask_position,
label ids,segement labels.the data sample was in data file.
### second step:
For simplicity, you can just run my run.py,then it will be 
my_pretraining>python my_pretraining.py --input_dir=D:\database\wiketext\sub_new\en_30528_hdf5 --config_file=base.json --output_dir=output --train_batch_size=12 --learning_rate=5e-05 --num_train_epochs=10 --max_steps=1000 --warmup_proportion=0.01 --resume_from_checkpoint=True --num_workers=4
