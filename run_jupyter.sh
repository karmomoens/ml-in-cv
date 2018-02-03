xhost +

docker run -it --privileged \
    --env="DISPLAY" \
    --network=host \
    -p 8888:8888 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $PWD/src:/media/local/src \
    -v $PWD/examples:/media/local/examples \
    -v $PWD/sprites:/media/local/sprites \
    -v /dev/video0:/dev/video0 \
    -v $PWD/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py \
    student-trip-lyon bash

xhost -
