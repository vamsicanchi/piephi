# Redhat 9 Universal Base Image
FROM redhat/ubi9:latest

# Tag
LABEL Name=piephi Version=0.0.1

# Redhat Apps
RUN yum --disableplugin=subscription-manager --skip-broken install sudo gcc gcc-c++ curl libcurl-devel autoconf automake libtool pkgconfig cmake make rsync -y
RUN dnf --disableplugin=subscription-manager --skip-broken install openssl openssl-devel openssh bzip2-devel libffi-devel mesa-libGL glibc krb5-libs -y
RUN yum --disableplugin=subscription-manager --skip-broken install unixODBC unixODBC-devel postgresql-odbc postgresql-plpython3 mariadb-connector-odbc -y
RUN yum --disableplugin=subscription-manager --skip-broken install libtiff libtiff-devel libpng libpng-devel libjpeg libjpeg-devel libjpeg-turbo-devel libwebp libwebp-devel -y
RUN yum --disableplugin=subscription-manager --skip-broken install ghostscript libreoffice nginx nodejs poppler poppler-qt5 poppler-utils -y
RUN yum --disableplugin=subscription-manager --skip-broken install tee sqlite sqlite-devel expat expat-devel libpq-devel -y
RUN yum --disableplugin=subscription-manager --skip-broken install python39 python3-devel python3-pip -y
RUN curl https://packages.microsoft.com/config/rhel/9/prod.repo | sudo tee /etc/yum.repos.d/mssql-release.repo
RUN sudo ACCEPT_EULA=Y yum install -y msodbcsql18

# Copy tar file into image and set destination as working location
RUN mkdir /usr/src/apps
COPY build/ /usr/src/apps

# Install Leptonica 1.83.1
RUN ls /usr/src/apps
RUN echo $(pwd)
RUN echo $(ls /usr/src/apps)
WORKDIR /usr/src/apps/leptonica-1.83.1
RUN cd /usr/src/apps/leptonica-1.83.1
RUN ./configure --disable-dependency-tracking
RUN make
RUN make install
RUN cp /usr/local/lib/pkgconfig/lept.pc /usr/lib64/pkgconfig/

# Install Tesseract 5.3.3
WORKDIR /usr/src/apps/tesseract-5.3.3
RUN ./autogen.sh
RUN export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
RUN export LD_LIBRARY_PATH=/usr/local/lib
RUN export PKG_CONFIG=/usr/bin/pkg-config
RUN LIBLEPT_HEADERSDIR=/usr/local/include ./configure --prefix=/usr/local/ --with-extra-libraries=/usr/local/lib
RUN make
RUN make install
RUN ldconfig

# Install Libpostal 1.1
WORKDIR /usr/src/apps/libpostal-1.1
RUN ./bootstrap.sh && ./configure --datadir=/usr/src/apps/postaldata --disable-data-download && make -j4 && make install && ldconfig

# Install PROJ 9.3.0
WORKDIR /usr/src/apps/proj-9.3.0
RUN mkdir /usr/src/apps/proj-9.3.0/build && cd /usr/src/apps/proj-9.3.0/build
RUN sudo cmake .
RUN sudo cmake --build .
RUN sudo cmake --build . --target install
RUN cp -r /usr/src/apps/proj-data-1.15 ${XDG_DATA_HOME}/proj

# Install GDAL 3.7.2
WORKDIR /usr/src/apps/gdal-3.7.2
RUN mkdir /usr/src/apps/gdal-3.7.2/build && cd /usr/src/apps/gdal-3.7.2/build
RUN sudo cmake .
RUN sudo cmake --build .
RUN sudo cmake --build . --target install

# Install Python 3.9 libraries
RUN cd
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# RUN pip3 cache purge
# COPY check.py .

# Command
CMD ["sleep", "infinity"]