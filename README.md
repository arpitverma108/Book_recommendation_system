Now run,
```bash
streamlit run app.py
```


# Streamlit app Docker Image Deployment

## 1. Login with your AWS console and launch an EC2 instance
## 2. Run the following commands

Note: Do the port mapping to this port:- 8501

```bash
sudo apt-get update -y

sudo apt-get upgrade

#Install Docker

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

```bash
git clone https://github.com/arpitverma108/Book_recommendation_system.git
```

```bash
docker build -t arpitverma108/book_recommendation_system:latest .

```

```bash
docker images -a  
```

```bash
docker run -d -p 8501:8501 arpitverma108/book_recommendation_system:latest

```

```bash
docker ps  
```

```bash
docker stop container_id
```

```bash
docker rm $(docker ps -a -q)
```

```bash
docker login 
```

```bash
ddocker push arpitverma108/stapp:latest

```

```bash
docker rmi arpitverma108/stapp:latest
```

```bash
docker pull arpitverma108/stapp
```