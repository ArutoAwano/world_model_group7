FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Install system dependencies for OpenCV and rendering
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
# gymnasium: for reinforcement learning environments
# moviepy: for saving videos
# imageio: for video processing
# wandb: for experiment tracking
# opencv-python: for image processing
# matplotlib: for plotting (if needed)

RUN pip install --no-cache-dir \
    gymnasium[classic_control] \
    moviepy \
    imageio \
    wandb \
    opencv-python \
    matplotlib \
    tqdm \
    pillow

# Set environment variables for non-interactive rendering if needed
ENV MUJOCO_GL=egl
RUN echo "export PS1='\[\e[1;36m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ '" >> /root/.bashrc

# Keep container running
CMD ["bash"]
