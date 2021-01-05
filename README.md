# image-classification
- Kaggle competition accuracy 0.91
- Kaggle link: https://www.kaggle.com/c/ift3395-6390-quickdraw/leaderboard
- Rank #1
- Team EP


# How to run:

- Move vgg_with_changing_top_layers.py in the same directory as **train.npz** and **test.npz**
- Important: For most effective way to run vgg_with_changing_top_layers.py, it is suggested to run this code in google colab, 
as this code was tested on that platform.
- Run:
```bash
$ python vgg_with_changing_top_layers.py
```
This will train the model and save the prediction for the test set in a file called **results.csv**
