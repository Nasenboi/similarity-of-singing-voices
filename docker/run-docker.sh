source ../.env

docker build --tag tensorflow-miniconda:latest .

docker run -it --name sov --gpus all -p 2718:2718 -v ${PWD}/../:/app -v ${DRIVE}:/data tensorflow-miniconda