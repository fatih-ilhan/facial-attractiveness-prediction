# facial-attractiveness-prediction

Fatih İlhan - 21401801

Selim Furkan Tekin - 21501391

# Installing

## Conda Run
After creating and activating your virtual environment install requirements

``
$ pip install -r requirements.txt
``


## Docker Run

*Build docker image* 


Go to `docker` file


```$ cd docker```

Build the image and go back to project directory

``$ docker build -t facial:v1 .``


``$ cd .. ``

Run docker container

``$ docker run -it --rm -v `pwd`:/workspace facial:v1 ``

Then inside container run `main.py`

``/workspace# python main.py``


# Running

Put the data files under `data` directory following this structure

```
.
+-- data
|   +-- test
|   +-- train
|   +-- validation
```

Further you can put sample images under `data/sample` folder


To run the code it is enough to write 

```
$ python main.py
```

you can specify arguments

```
$ python main.py --device GPU --slot 0 --overwrite 1
```

where overwrite used to overwrite previous results.

Modify `config.py` to change experiment configuration.

Experiment outputs will be written under `results` folder.
