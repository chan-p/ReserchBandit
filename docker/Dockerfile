FROM arwineap/docker-ubuntu-python3.6

MAINTAINER Tomonori Hayashi "hyshtmnr6238@gmail.com"

RUN apt-get update && apt-get -y upgrade && apt-get install -y build-essential libssl-dev libreadline-dev zlib1g-dev language-pack-ja
RUN apt-get -y install openssh-server ufw curl
RUN mkdir /var/run/sshd
RUN bash -c 'echo "root:root" | chpasswd'

CMD /usr/sbin/sshd -D && tail -f /dev/null
EXPOSE 22

RUN mkdir DockerWorking
RUN cd DockerWorking

RUN apt-get install git-man less liberror-perl libpopt0 rsync git
RUN git clone https://github.com/chan-p/ReserchBandit.git

RUN ls
RUN pip install -r ./ReserchBandit/docker/packages_requirements.txt
