# BUILD STAGE

FROM public.ecr.aws/lambda/python:3.9 AS build

ARG API_KEY_INPUT

ENV API_KEY $API_KEY_INPUT
ENV SURPRISE_DATA_FOLDER ./data

COPY app ./app

COPY movies_data.pkl .

COPY main.py .
COPY data.py .
COPY config.py .
COPY rs_models ./rs_models
COPY data ./data

COPY requirements.txt  .
RUN yum install gcc -y
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# FINAL STAGE
FROM public.ecr.aws/lambda/python:3.9

ARG API_KEY_INPUT

ENV API_KEY $API_KEY_INPUT
ENV SURPRISE_DATA_FOLDER ./data

# Copy only the necessary files from the build stage
COPY --from=build "${LAMBDA_TASK_ROOT}" "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY app/api.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "api.handler" ]