# fl_spoofing_detection
A simulation for a federated learning system that detects order spoofing when given a data batch. This project is part of my diploma (bachelor) thesis.

## Data Processing and Feature Engineerung
This project uses L3 and L2 Data from Databento (https://databento.com). Specifically, it uses MBO and MBP-10 data. Features are split into present and future sets (features, that can only be calculated afterwards). The features are calculated to "anchor events" only, which are bigger orders (twice the daily average size). 

## Weak Labeling
A major issue and weakness of the thesis is the fact, that there are no spoofing labels. I created weak labels using "labeling functions" from Snorkel (https://github.com/snorkel-team/snorkel). These use the afterwards-features.

## Learning with Baseline XGBoost and FedXGBoost
The present features are used for training and testing. The baseline config is being validated in a TSCV-loop. The federated simulation system is provided by Flower's XGBoost-quickstart (https://github.com/flwrlabs/flower). There have been made some major changes though, check in the commit history for more.

## Results
The thesis is about the question, whether federated learning can be used to identify financial crimes (with market spoofing as an example). Conducting the empirical study showed that federated learning performs similarly to the baseline model (though I didn't investigate into the usage of privacy-preserving methods). Extensive discussion of the results, the identification and evaluation of weakness was necessary.

## Other
You might notice two different venv's in the .gitignore. This is because I realized too late, that Tensorflow and Tensorflow Federated has no support after Python 3.11. It is recommended to make one virtual environment with Python Version 3.11

I utilized AI in this project for first ideas and code error fixing, mainly ChatGPT and Claude.
