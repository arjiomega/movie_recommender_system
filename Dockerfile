FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.

ARG API_KEY_INPUT

ENV API_KEY $API_KEY_INPUT

COPY app ./app

COPY movies_data.pkl .

COPY main.py .
COPY data.py .


COPY requirements.txt  .
RUN yum install gcc -y
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY app/api.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "api.handler" ]