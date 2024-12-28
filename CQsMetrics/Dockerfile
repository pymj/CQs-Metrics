FROM python:3.8.5
# Set the working directory in the container
RUN pip install spacy
RUN pip install nltk
RUN python -m spacy download en_core_web_sm
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')" ]
RUN pip install --upgrade pip
# ENV PATH="/app:${PATH}"
COPY CQsMetrics/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# COPY CQsMetrics/model/t5  /app/model/t5
COPY CQsMetrics/process_data.py /app/process_data.py
COPY CQsMetrics/data /app/data
COPY CQsMetrics/config.py /app/config.py
COPY CQsMetrics/scope.py /app/scope.py
COPY CQsMetrics/relevance.py /app/relevance.py
COPY CQsMetrics/themes.py /app/themes.py
COPY CQsMetrics/main.py /app/main.py
COPY CQsMetrics/answerability.py /app/answerability.py
# COPY AgOCQs/inputText /app/inputText/request
WORKDIR /app
#COPY . /app

RUN ls
ENTRYPOINT [python, "main.py"]



