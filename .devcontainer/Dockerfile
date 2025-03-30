From firedrakeproject/firedrake

ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID
RUN userdel -r ubuntu
RUN <<EOF
    addgroup --gid ${REMOTE_GID} ${REMOTE_USER}
    adduser --disabled-password --uid ${REMOTE_UID} --gid ${REMOTE_GID} ${REMOTE_USER}
EOF

# Give remote user sudo access
RUN apt-get update && apt-get install -y sudo && \
  echo "${REMOTE_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${REMOTE_USER}

USER ${REMOTE_USER}
