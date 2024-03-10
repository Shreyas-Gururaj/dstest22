# dstest22
This project repository contains the scripts for training and evaluating two modules (i.e. Item bundle recommendation and "UnitPrice" prediction) for the given input eCommerce dataset. Each of the modules can be independently containerized and executed using the Docker File provided inside the  repository.

### The below command is used to clone the repo :
```
git clone https://github.com/Shreyas-Gururaj/dstest22.git
```
### 1. FASTAPI-based Inference module set-up :
```
cd dstest22/
docker build --build-arg MODE="FASTAPI" -t image_api .
docker run -ti --name cont_api -p 8000:8000 image_api
```

#### 1.2. Open the local host in the browser: "http://127.0.0.1:8000/docs"

#### 1.3. Provide the following JSON example input, these are randomly selected "CustomerID" / "StockCode" from the test dataset:
```
{
  "CustomerID": [16233, 12355, 13004, 14133, 17354, 13229, 14234, 16365, 16168, 17175],
  "StockCode": ["23512", "22693", "23399", "23144", "22049", "21232", "21642", "22979", "22423", "21735"]
}
```

#### 1.4. Check the response

### 2. Regression module set-up :
```
cd dstest22/
docker build --build-arg MODE="regression" -t image_train_regression .
docker run -ti --name cont_train_regression image_train_regression
```

### 3. NCF to train the model for recommendation module set-up :
```
cd dstest22/
docker build --build-arg MODE="train_reccomendation" -t image_train_reccomendation .
docker run -ti --gpus all --name cont_train_reccomendation image_train_reccomendation
```
### 4. Training and validation logs:

#### 4.1. Recommendation model NCF train logs: "/app/train_eval.log"
#### 4.2. Regression train logs: "/regression/regression.log"

### 5. References and project details :
- Specific details related to data pre-processing, feature engineering, train/test split, model architecture, evaluation metric, and answers to the questions related to business impact are captured in a separate document "dstest22/Task_specific_details.pdf"
- Link to check the training loss curve for the NCF model used for the Recommendation task: "https://wandb.ai/shreyas1995/NeuralCF_Conrad?nw=nwusershreyas1995"
- Neural Collaborative Filtering (NCF) paper: "https://arxiv.org/abs/1708.05031"
