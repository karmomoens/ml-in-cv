xhost +

docker run -it --rm --privileged \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --network=host \
    -v "$PWD:/media/local/student-trip-lyon" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PWD/pyCharmConfig:/root/.PyCharmCE2017.1" \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v /dev/video0:/dev/video0 \
    -v /dev/snd:/dev/snd \
    bapha/student-trip-lyon bash

xhost -
