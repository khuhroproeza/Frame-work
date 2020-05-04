Getting Started
Anomaly Detection ReadMe
Anomaly Detection Framework

Algorithms
AutoEncoder
LSTM Compressed AutoEncoder
One Class SVM
SVM
Datasets
Bohren
Drilling Dataset from IFW. Data recorded from two different types of run. Normal run classified as 0, (drilling Process under normal conditions) Abnormal run classified as 1 (drilling Process with Chatter occurance)

HPC Performance
Data tken from target system called "D.A.V.I.D.E", an energy efficient supercomputer developed by E4 Computer Engineering and hosted by CINECA in Bologna, Italy. Data aggregated in 5-minutes long intervals. There are around 166 metrics (our features), i.e. core loads, temperatures, fan speed, power consumptions, etc. Normal Run classified as 0. Forced Anomalies related to the performance of the HPC classified as 1

HPC Powersave
Another branch of HPC dataset, where forced anomalies are focused on powersaving. Normal run classified as 0 and forced anomalies as 1

Conditioning
This dataset contains a timestamp and 25 signals of 8 different components. For example, L1 could be the current of a stepper motor and L2 the speed of that stepper motor. this data shows the process of a whole production plant. The process was run till failure. Thus intial readings of the dataset are classified as 0 and end failure classified as 1 (last 500 readings to be certain as per the author of the dataset).

TEP
The Tennessee Eastman process is a typical industrial process that consists of five main process units: a two-phase reactor where an exothermic reaction occurs, a separator, a stripper, a compressor, and a mixer. This is a nonlinear open-loop unstable process that has been used in many studies as a case study for plant-wide control, statistical process monitoring, sensor fault detection, and identification of data-driven network models. 0 Classifies Normal 1 classifies Abnormal data

Running the Program
Editing Bash file named "exprun.sh"
Setting input argument to either Option=('Show')or Option=('Train')
Setting in number of Runs to perform
Running the bash file from the main folder directory; eg: "./exprun.sh" via terminal
Framework Setup
Framework
Converters
Contains Converter python file to divide data as per the requirement of the aloggorithm

Datasets
Contains all the datasets

Measures
Python functions to compute the measures used in the framework

Model
Contains all the algorithms used in the framework

savedresult
Contains all the saved results as pkl file saving, measures and names of dataset + algorithm

static
Contains CSS file for Flask Output Result html page

Templates
Contains all the html pages used in the framework

Exprun.sh
Main Bash file to start the program

Flaskblog.py
Flask file to output and render all the pages

Main.py
Main file which files all the algorithms and saves the pickle files

Run.py
Run file which accepts arguements from the bash file to either run or show framework

SqlCreater
Functions to save result as SQL and to create different dictionaries for the use of flask page rendering

Packages Installed
absl-py==0.8.1
Click==7.0
converters==0.0.1
astor==0.8.0
cycler==0.10.0
Cython==0.29.14
Flask==1.1.1
gast==0.3.2
google-pasta==0.1.7
grpcio==1.24.3
h5py==2.10.0
itsdangerous==1.1.0
Jinja2==2.10.3
ajoblib==0.14.0
Keras==2.3.1
astor==0.8.0
astor==0.8.0
astor==0.8.0
astor==0.8.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
kiwisolver==1.1.0
Markdown==3.1.1
MarkupSafe==1.1.1
matplotlib==3.1.1
mysql-connector-python==8.0.18
numpy==1.17.3
pandas==0.25.2
patsy==0.5.1
pkg-resources==0.0.0
property-cached==1.6.3
protobuf==3.10.0
pyparsing==2.4.2
python-dateutil==2.8.0
pytz==2019.3
PyYAML==5.1.2
scikit-learn==0.21.3
scipy==1.3.1
seaborn==0.9.0
six==1.12.0
statsmodels==0.10.1
tensorboard==1.14.0
tensorflow==1.14.0
tensorflow-estimator==1.14.0
termcolor==1.1.0
torch==1.3.1
Werkzeug==0.16.0
wrapt==1.11.2
