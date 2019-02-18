FROM nvcr.io/nvidia/pytorch:18.06-py3
ENV http_proxy=http://gateway.zscaler.net:9480
ENV https_proxy=http://gateway.zscaler.net:9480
ENV ftp_proxy=http://gateway.zscaler.net:9480


RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate pytorch-py3.6 && conda install -c rdkit rdkit 
RUN echo "source activate pytorch-py3.6" >~/.bashrc
RUN conda install pytorch=0.3.1  torchvision cuda90 -c pytorch
#RUN conda install -c rdkit rdkit
RUN pip install joblib
#RUN pip install --upgrade pip  && pip install jupyter
RUN pip install pandas
RUN pip install scipy matplotlib sympy nose
RUN pip install -U scikit-learn
RUN pip install mlxtend
RUN pip install h5py
RUN pip install future
EXPOSE 8888

WORKDIR /work
CMD ["jupyter", "notebook", "--ip", "*","--allow-root"]





