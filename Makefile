help:
	@cat Makefile

NAME="docker-sample"
SRC?="$(shell pwd)/" #src directory
DATA?="$(shell pwd)/data/" #data direcotry
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=9.0
CUDNN_VERSION?=7
TEST=tests/

build:
	docker build -t tensorflow --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --name $(NAME) --env KERAS_BACKEND=$(BACKEND) tensorflow bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --name $(NAME) --env KERAS_BACKEND=$(BACKEND) tensorflow ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --name $(NAME) --net=host --env KERAS_BACKEND=$(BACKEND) tensorflow

test: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --name $(NAME) --env KERAS_BACKEND=$(BACKEND) tensorflow py.test $(TEST)

