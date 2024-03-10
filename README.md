# dstest22
This project repository contains the scripts for training and evaluating two modules (i.e. Item bundle recommendation and "UnitPrice" prediction) for the given input eCommerce dataset. Each of the modules can be independently containerized and executed using the Docker File provided inside the  repository.

### The below command is used to clone the repo
```
git clone https://github.com/Shreyas-Gururaj/dstest22.git
```
### 1.1 FASTAPI-based Inference module set up
```
cd dstest22/
docker build --build-arg MODE="FASTAPI" -t image_api .
docker run -ti --name cont_api -p 8000:8000 image_api
```
### 1.2 Provide input Json objects pairs of "CustomerID" and "StockCode" and the response is a Json object containing the top-5 personalized item recommendations with their respective descriptions and sum_bundle_price
```
example input Json :
{
  "CustomerID": [16233, 12355, 13004, 14133, 17354, 13229, 14234, 16365, 16168, 17175],
  "StockCode": ["23512", "22693", "23399", "23144", "22049", "21232", "21642", "22979", "22423", "21735"]
}
