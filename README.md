# Sagemaker-amazon
A thyroid disease detection, Amazon Sagemaker using Scikit-learn Pipeline (StandardScalar &amp; SVM)
# Implementaion
in the implementing of this smart system we used two models. First the standard scalar model for prepossessing the input data. Second, the SVM model to train a model based on our data. We used the Amazon Sagemkaer as a helpful tool to be able to train our model. Two python scripts have been written and have been passed as an entry point to the Sickit learn library (we did not use the pre-built algorithm provided in Sagemaker and we create our scripts). In the training section, the models are saved in Amazon S3 to be able to use them. Two models are passed to a pipeline and the final model is deployed to the endpoint. To be able to connect to the endpoint, we used an API getaway in Amazon. The get request will be sent to a lambda function. Lambda function will provide the information in a suitable structure for the endpoint. The InvokEndpoint method is used in the lambda function and the response will be returned. All of the learning section is done based on the jupyter notebook instance. By using Sagemaker we can customize the algorithms like SVM with different types of kernels.
# How to run

 1. Upload the all the files in to the AWS notebook instance.
 2. Put the S3 bucket url into the notebook instance
 3. Run the notebook instance.

# Results

Measurments gathered from the model is summrized as follows:
 - Accurecy: 96.29%
 - Precison: 83.95%
- Recall: 75.55%
- F1-score : 79.53% 
