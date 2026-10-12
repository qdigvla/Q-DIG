docker build -t qd-robotics-safety .

docker run -it \
    --gpus '"device=0,1,2,3"' \
    --network=host -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/tmp/.Xauthority \
    -v /etc/vulkan/icd.d/nvidia_icd.json:/tmp/nvidia_icd.json \
    -e VK_ICD_FILENAMES=/tmp/nvidia_icd.json \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e XAUTHORITY=/tmp/.Xauthority \
    -v /usr/share/nvidia:/usr/share/nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /home/ssrikant/Documents/openvla-oft/checkpoints:/checkpoints \
    qd-robotics-safety:save bash

# docker run -it \
#     --gpus '"device=0,1,2,3"' \
#     --network=host \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     -v $HOME/.Xauthority:/root/.Xauthority:rw \
#     -e XAUTHORITY=/root/.Xauthority \
#     -v /usr/share/nvidia:/usr/share/nvidia \
#     -v /etc/vulkan/icd.d/nvidia_icd.json:/tmp/nvidia_icd.json \
#     -e VK_ICD_FILENAMES=/tmp/nvidia_icd.json \
#     -e NVIDIA_DRIVER_CAPABILITIES=all \
#     qd-robotics-safety bash
