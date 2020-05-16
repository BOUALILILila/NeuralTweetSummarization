# Neural models for extractive tweet summarization
This repository contains the code for reproducing the experiments of our CIRLE 2020 paper: "Retrospective Tweet Summarization: Investigating NeuralApproaches for Tweet Retrieval".

## General pipeline
We assume that a summary is a set of the top non-redundant tweets that are relevant to the user’s interest topic. First, we attempt to reduce the semantic gap between queries and tweets using distributed representations (Word2vec, GloVe and Fasttext). Then, we investigate deep neuronal models from the [matchzoo library](https://github.com/NTMC-Community/MatchZoo) (DRMM, ACR_II, DUET and Matchpyramid) – that are capable of learning a relevancefunction from the text inputs without further hand crafted features – to retrieve candidate relevant tweets. This overcomes the ineffectiveness of term-frequency based models for short text like tweets. As the resulting candidate list of tweets may contain redundant information, clusters of similar tweets are created using their representations in the latent space. The summary is then constituted from the representative tweet of each cluster.

![](images/model.png)

## Web Application 
The ```RestAPP``` contains the code of the tweet sumarization api, the ```WebClient``` is a simple web interface for tweet summarization given the topic introduced by the user.
![](images/web_app.png)

