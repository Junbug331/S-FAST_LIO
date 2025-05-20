#ros noetic
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=nointeractive


RUN apt update
RUN apt install curl gnupg2 lsb-release -y
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-perception \
    && rm -rf /var/lib/apt/lists/

RUN apt-get update && apt-get install -y ca-certificates gpg wget &&\
    test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null &&\
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null &&\
    apt-get update &&\
    test -f /usr/share/doc/kitware-archive-keyring/copyright || rm /usr/share/keyrings/kitware-archive-keyring.gpg &&\
    apt-get install -y kitware-archive-keyring &&\
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal-rc main' | tee -a /etc/apt/sources.list.d/kitware.list >/dev/null &&\
    apt-get update &&\
    apt-get install -y cmake

RUN apt install -y libboost-dev ninja-build

# Install prerequisites for add-apt-repository
RUN apt-get update && \
    apt-get install -y software-properties-common

# install git
RUN add-apt-repository ppa:git-core/ppa &&\
    apt-get update &&\
    apt-get install -y git

# eigen
WORKDIR /root/
RUN git clone https://gitlab.com/libeigen/eigen.git
WORKDIR /root/eigen
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE="Release" -GNinja && cmake --build build && cmake --install build
RUN rm -rf /root/eigen

# Sophus
WORKDIR /root/
RUN git clone https://github.com/strasdat/Sophus.git
WORKDIR /root/Sophus
RUN cmake -S . -B build -DUSE_BASIC_LOGGING=ON -DCMAKE_BUILD_TYPE="Release" -GNinja && cmake --build build && cmake --install build
RUN rm -rf /root/Sophus

#install gtsam
RUN apt-get update \
    && apt install -y software-properties-common \
    && add-apt-repository -y ppa:borglab/gtsam-release-4.0 \
    && apt-get update
RUN apt install -y libgtsam-dev libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen && \
    ldconfig

#rviz install
RUN apt update && apt install -y ros-noetic-rviz


#install ros package
RUN apt install -y ros-noetic-xacro \
    ros-noetic-robot-state-publisher \
    ros-noetic-robot-localization

RUN apt install -y ros-noetic-tf-conversions
RUN apt-get install -y libfmt-dev

#install Livox SDK
WORKDIR /root/
RUN apt install -y build-essential
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && \
    find Livox-SDK -type f -name CMakeLists.txt \
         -exec sed -i 's/^cmake_minimum_required(VERSION .*)/cmake_minimum_required(VERSION 3.5)/' {} \; && \
    mkdir -p Livox-SDK/build && cd Livox-SDK/build && \
    cmake .. && make && make install && ldconfig

WORKDIR /root/
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
    find Livox-SDK2 -type f -name CMakeLists.txt \
         -exec sed -i 's/^cmake_minimum_required(.*)/cmake_minimum_required(VERSION 3.5)/' {} \; && \
    mkdir -p Livox-SDK2/build && cd Livox-SDK2/build && \
    cmake .. && make -j && make install && ldconfig

#install livox ros driver
RUN cd /root && mkdir catkin_ws && mkdir catkin_ws/src && \ 
    cd catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    find livox_ros_driver -type f -name CMakeLists.txt \
         -exec sed -i 's/^cmake_minimum_required(.*)/cmake_minimum_required(VERSION 3.5)/' {} \;

RUN sed -i \
  's|^cmake_minimum_required(.*)|cmake_minimum_required(VERSION 3.5)|' \
  /usr/src/googletest/CMakeLists.txt

RUN sed -i \
  's|^cmake_minimum_required(.*)|cmake_minimum_required(VERSION 3.5)|' \
  /usr/src/googletest/googlemock/CMakeLists.txt

RUN sed -i \
  's|^cmake_minimum_required(.*)|cmake_minimum_required(VERSION 3.5)|' \
  /usr/src/googletest/googletest/CMakeLists.txt

# Override the default CMakeLists.txt in src/
WORKDIR /root/catkin_ws/src
RUN rm -f CMakeLists.txt && \
    echo 'cmake_minimum_required(VERSION 3.5)' > CMakeLists.txt && \
    echo 'find_package(catkin REQUIRED)'    >> CMakeLists.txt && \
    echo 'catkin_workspace()'               >> CMakeLists.txt


# Build the workspace
WORKDIR /root/catkin_ws
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && catkin_make && \
                   source devel/setup.bash"


#entrypoint setup
RUN echo '#!/bin/bash' > /ros_entrypoint.sh
RUN echo 'set -e' >> /ros_entrypoint.sh
RUN echo ' ' >> /ros_entrypoint.sh
RUN echo '# setup ros environment' >> /ros_entrypoint.sh
RUN echo 'source "/opt/ros/noetic/setup.bash"' >> /ros_entrypoint.sh
RUN echo 'exec "$@"' >> /ros_entrypoint.sh

RUN chmod +x /ros_entrypoint.sh
RUN sed -i "5i source \"/opt/ros/noetic/setup.bash\"" /ros_entrypoint.sh
RUN bash -c "echo source /opt/ros/noetic/setup.bash\ >> /root/.bashrc"


ENTRYPOINT ["/ros_entrypoint.sh"]
