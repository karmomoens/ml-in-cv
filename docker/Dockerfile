FROM ubuntu:artful

RUN apt-get update && \
    apt-get install -y build-essential \
                       pkg-config \
                       libopencv-dev \
                       libjpeg8-dev \
                       libtiff5-dev \
                       libpng-dev \
                       libavcodec-dev \
                       libavformat-dev \
                       libswscale-dev \
                       libv4l-dev \
                       libxvidcore-dev \
                       libx264-dev \
                       libatlas-base-dev \
                       gfortran \
                       python3 \
                       python3-pip \
                       cmake \
                       libgtk-3-dev \
                       libboost-all-dev \
                       ssh \
                       gstreamer1.0-plugins-*


COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install Pycharm
RUN wget -P /tmp/ https://download.jetbrains.com/python/pycharm-community-2017.1.tar.gz && \
    tar xvzf /tmp/pycharm-community*.tar.gz -C /tmp/ && \
    rm /tmp/pycharm-community*.tar.gz && \
    mv /tmp/pycharm-community* /opt/pycharm-community && \
    ln -s /opt/pycharm-community/bin/pycharm.sh /usr/local/bin/pycharm && \
    ln -s /opt/pycharm-community/bin/inspect.sh /usr/local/bin/inspect

RUN apt-get install zip unzip
# Install OpenCV 3.2
WORKDIR /
RUN wget https://github.com/Itseez/opencv/archive/3.2.0.zip \
&& wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip \
&& unzip 3.2.0.zip \
&& unzip opencv_contrib.zip \
&& mkdir /opencv-3.2.0/cmake_binary \
&& cd /opencv-3.2.0/cmake_binary \
&& cmake \
  -DBUILD_NEW_PYTHON_SUPPORT=ON \
  -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-3.2.0/modules \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make -j4 \
&& make install \
&& ldconfig \
&& rm /3.2.0.zip \
&& rm /opencv_contrib.zip \
&& rm -r /opencv-3.2.0 \
&& rm -r /opencv_contrib-3.2.0

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

ADD $PWD /media/local/
ENV PYTHONPATH=/media/local/
WORKDIR /media/local

CMD bash
