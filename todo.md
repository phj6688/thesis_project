Thesis Workflow
---------------

### Choose the datasets for the experiment:

*   PC
*   CR
*   SUBJ
*   TREC
*   SST-2
*   Kaggle Medical
*   Ohsumed
*   AG News

### Select the data augmentation methods to be used:

*   EDA
*   AEDA
*   WordNet
*   Backtranslation
*   Clare
*   Checklist
*   Data Boost

### Decide on the models to be used for the experiment:

*   RNN (bidirectional LSTM)
*   CNN
*   BERT

### Prepare the data:

*   Load the datasets into memory.
*   Preprocess the datasets as necessary (e.g., remove punctuation, stop words, etc.).
*   Divide the training set into sub-samples of 10%, 20%, 50%, and 100%.

### Train the models on the original training data and test on the testing data:

*   Train an RNN (bidirectional LSTM) model.
*   Train a CNN model.
*   Train a BERT model.
*   Test the models on the testing data and record the results.

### Augment the training data with one, two, and four samples per data point:

*   Apply each of the selected data augmentation methods to each training data sub-sample.
*   Combine the augmented sub-samples with the original sub-sample.
*   Train the models on the augmented training data and test on the testing data:

*   Train an RNN (bidirectional LSTM) model on the augmented data.
*   Train a CNN model on the augmented data.
*   Train a BERT model on the augmented data.
*   Test the models on the testing data and record the results.

### Analyze the results:

*   Compare the performance of the models trained on the original data to the models trained on the augmented data.
*   Analyze the impact of the different data augmentation methods on the model performance.
*   Use common metrics such as accuracy, f1, and auc to evaluate the performance.
*   Visualize the augmented data using tsne, umap, and pca to compare it to the original data.

### Write up the results:

*   Summarize the experiment.
*   Report the results of the experiment.
*   Discuss the implications of the results.
*   Make recommendations for future research.