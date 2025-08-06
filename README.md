# Emulating Radiative Transfer in Astrophysical Environments

This repository is the official implementation of [Emulating Radiative Transfer in Astrophysical Environments](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

Radiative transfer - cornerstone - ... - gifs und plots die einfach Vorhersage zeigen, Ergebnisplots dann unten

## Installation

To clone the git:

```bash
git clone https://github.com/your-anonymous-id/Astro_RT.git
cd Astro_RT
```

To install requirements:

```bash
pip install -r requirements.txt
```

sourcen + wie an dataset kommen (Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)


## Training

In our paper we consider two scenarios in which we want to emulate Radiative Transfer. ....

To train the steady-state model, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

To train the steady-state model, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

sagen, was in beiden Dateien vohanden, etc. was man verstellen kann, sagen, dass auf die in architectures (link hinterlegen) defineirten architekturen zugreifen, Optuna etc

bei architecturen nochmal die Referenz von ufno citen

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation
-> soll ich das extra machen, wenn ja dann bei git auch evaluate.py hinzufügen

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

The two pre-trained models we present in our paper can be found in the folder surrogate models (mit link hinterlegen).
ufno_3d.eqx (hinterlegen) und ufno_3d_time.eqx (hinterlegen vorstellen)

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Following plots show a comparison of the predictions of surrogate models and numerical reference for a random sample from the test set

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. -> d.h. ich sollte evaluate vermutlich drinhaben, aber da einfach nur plots mit model machen bzw vllt nochmal test loss

Tabelle mit accuracy relativ und mse vllt machen für beide models, vllt auch hyperparameter


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
