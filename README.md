# facial-attractiveness-prediction

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
