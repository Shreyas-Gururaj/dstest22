# dstest22
This project repository contains the scripts for training and evaluating two modules (i.e. Item bundle recommendation and "UnitPrice" prediction) for the given input eCommerce dataset. Each of the modules can be independently containerized and executed using the Docker File provided inside the  repository.

### The below command is used to clone the repo
```
git clone https://github.com/Shreyas-Gururaj/dstest22.git
```
# 1.1 FASTAPI-based Inference module set up
```
cd dstest22/
docker build --build-arg MODE="regression" -t image_reg .
docker run -ti --name cont_reg image_rec
```
