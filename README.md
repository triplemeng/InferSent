# InferSent
The repo is an implementation of the paper "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" (a.k.a. InferSent) by Alexis Conneau et. al. 

To prepare the data:

1. bash ./data.sh  
It will download the snli zipped data and uncompress it to ./data/snli_1.0 folder with train/validation/test files.
It will also download the glove.840B.300d.zip file and unzip it to glove.840B.300d.txt, which provides word embedding.

2. based on the words available in training dataset, build vocalbulary and embedding set for training purpose, and then 
reformat the train/validation/test data with each word as indexed in the vocalbulary.  
cd ./code  
python prepare_data.py  

3. pad the data set before give it as input to the model.  
python component.py  

(Note: component.py contains two encoders: one is DAN (deep averaging neural nets), the other is BiLSTM with max-pooling (which is the one with the best performance as recommended in the paper). One could easily add their own version of encoders and experiment with it.)

4. run the model  
python models.py  

usage: models.py [-h] [-b BATCH_SIZE] [-c ENCODER] [-e EPOCHS]
                 [-v ENCODED_DIM] [-n OPTIMIZATION] [-r RESUME]

Parameters for building the model.

optional arguments:  
  -h, --help            show this help message and exit  
  -b BATCH_SIZE, --batch_size BATCH_SIZE  
                        training batch size  
  -c ENCODER, --encoder ENCODER  
                        encoder type  
  -e EPOCHS, --epochs EPOCHS  
                        epochs for training  
  -v ENCODED_DIM, --encoded_dim ENCODED_DIM  
                        embedding size for compressed vector  
  -n OPTIMIZATION, --optimization OPTIMIZATION  
                        optimization algorithm  
  -r RESUME, --resume RESUME  
                        pick up the latest check point and resume  
                        
(Note: as in my other repo(https://github.com/triplemeng/hierarchical-attention-model), I used and modified r2rt's code in generating shuffled batch samples: https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)                    

Due to the limited resources, I use 100 as BiLSTM's hidden vector size,
and the batch size of 128. One should be able to adjust it accordingly to
achieve better accuracy. (the paper recommended the batch size of 64)
Based on my experiments, the model achieved 0.77 as accuracy on SNLI after 12
epochs with encoded_dim of 512 and optimiazation algorithm as 'sgd'.  

I didn't try the resulted sentence embeddings in any downstream NLP
tasks. I provided the code for generating sentence embeddings based on
trained models.  

python sentence_encoder.py

will ask you for the input sentence, and then generate the encoded
representation for it. 


