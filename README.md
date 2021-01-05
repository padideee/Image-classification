# image-classification
Kaggle competition accuracy 0.91, rank #1, team EP

# How to run vgg_with_changing_top_layers.py


- Move vgg_with_changing_top_layers.py in the same directory as **train.npz** and **test.npz**
- Important: For most effective way to run vgg_with_changing_top_layers.py, it is suggested to run this code in google colab, 
as this code was tested on that platform.
- Run:
```bash
$ python vgg_with_changing_top_layers.py
```
This will train the model and save the prediction for the test set in a file called **results.csv**
