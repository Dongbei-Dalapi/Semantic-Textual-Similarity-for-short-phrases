# Semantic-Textual-Similarity-for-short-phrases
an implementaion of the model detects if the two job titles are similar. Job titles are shorter than usual sentence, so we use character level embeddings instead of word level embeddings.

## Data
The training data is stored in train_data.csv where the similar titles are store in the same row. We create titles pair by linking titles from different rows and same row (data_process.py). The training data is splited and 30% of it is used as testing data. \ 

Additional testing data can be added to test_data.txt where the titles are separate by comma.

## Siamese LSTM
We use character-level bidirectional LSTM’s with a Siamese architecture

## Usage
```
$ pip3 install -r requirements.txt 
$ cd {project_folder}
```
### Training
```
$ python main.py [options/defaults]

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         number of training epochs (default: 50)
  --batch_size BATCH_SIZE
                        batch Size (default: 64)
  --train               train the model or not. If not, do evaluaion only (default: False).
  --save SAVE           specify the model name to save (default: SiameseLSTM.h5)
                    

```
### Performance
- Evaluation performance : similarity measure for 1173 pairs
| ---------- | ---- |
| Accuracy   | 1.0  |
| Precision  | 1.0  |
| Recall     | 1.0  |
| F1 Score   | 1.0  |


### Reference
[Learning text similarity with siamese recurrent networks](https://duckduckgo.com)

