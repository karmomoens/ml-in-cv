xhost +

docker run -it --rm --privileged \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --network=host \
    -p 8888:8888 \
    -p 8554:8554 \
    -p 8555:8555 \
    -p 8666:8666 \
    -v "$PWD:/media/local/ml-in-cv" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    -v /dev/video2:/dev/video2 \
    -v /dev/video3:/dev/video3 \
    -v /dev/video4:/dev/video4 \
    -v /dev/video5:/dev/video5 \
    -v /dev/snd:/dev/snd \
    -v $PWD/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py \
    bapha/student-trip-lyon jupyter notebook

xhost -
