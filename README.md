# Mnist_with-API-and-Docker
End to End Digit Recognizer.

Steps:

Download data:
```
python digit_recognizer/datasets/mnist_dataset.py 
```
Run Training:
Sample Configuration
```
python training/run_experiments.py --save "{\"dataset\": \"MNISTDataset\", \"model\": \"MnistModel\", \"network\": \"mlp\", \"train_args\":{\"batch_size\":128, \"epochs\":1}}"
```


Deploy on Flask
```
python api/app.py
```

Optional
Deploy using Docker

Optional Serverless Deployment on AWS Lambda

