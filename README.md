# docker_sample
nvidia-docker2 sample project

## env
- cuda9.0
- cudnn7
- python3.6
  - tensorflow-gpu
  - keras(backend tensorflow)

# how to use
1. `Makefile

- NAME:container name
- SRC:src path
- DATA:dataset path

2. `make bash`

3. `pip3 install -r requirements.txt`

4. `python titanic_sample.py`