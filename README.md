## Plant Disease Detection

## Inference Demo

https://github.com/user-attachments/assets/05a58353-82ef-4821-820f-fab68c3b12d3

## Docker Hub

You can find the [Plant Disease Detection Docker Image](https://hub.docker.com/r/yoda228/plant-disease-detection) on Docker Hub.

![Plant Disease Detection](plant-disease-detection-Docker-Image-Docker-Hub.png)

## How to Run

### On CPU:

`docker run -d -p 8001:8001 -p 7860:7860 --name plant-disease-app yoda228/plant-disease-detection:latest`

### On GPU:

`docker run -d -p 8001:8001 -p 7860:7860 --gpus all --name plant-disease-app-gpu yoda228/plant-disease-detection:latest`
