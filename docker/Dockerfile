FROM nvcr.io/nvidia/tritonserver:24.01-py3

ARG USERNAME=triton-server

RUN apt-get update \
    && apt-get install -y sudo cmake rapidjson-dev \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkhomedir_helper ${USERNAME}

RUN pip install --upgrade pip
RUN pip install tritonclient[http]

# 🔑 PATH 영구 설정
ENV PATH="/opt/tritonserver/bin:${PATH}"
ENV PYTHONPATH="/opt/tritonserver/backends/python:${PYTHONPATH}"

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

USER ${USERNAME}

WORKDIR /repository

CMD ["sleep", "infinity"]