FROM registry-vpc.cn-shanghai.aliyuncs.com/modelscope-repo/modelscope:fc-deploy-common-v17

WORKDIR /usr/src/app

COPY . .

RUN pip install -U transformers

CMD [ "python3", "-u", "/usr/src/app/app.py" ]

EXPOSE 9000
