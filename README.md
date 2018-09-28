# Name-Gender-Classification
It is a simple project to predict the gender(Male,Female) of Indian Names. I have used various Deep Learning model to get the things done.
### About Dataset
The dataset I have used is present [here](https://gist.github.com/mbejda/7f86ca901fe41bc14a63) and [here](https://gist.github.com/mbejda/9b93c7545c9dd93060bd)

The dataset is balanced with a total of 15382 Female and 14845 male names. There were few noises like name with address, name with son of/ daughter of etc.

### Model
I have used Keras as my deep learning framework for the project.
* First model I have tried was a Simple MLP with input as one hot encoded names as input for testing purpose. The acuracy I got on test set was **79.23%**.

* The second model I have tried was Deep Character level LSTM with `30,70 and 30` units. The accuracy I got was **91.28%**.
