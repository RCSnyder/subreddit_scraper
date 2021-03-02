## cd into ./backend

## set up a python venv
> python3 -m venv env

## activate it
windows:
> env/Scripts/activate

#### alternative for windows
> pip install virtualenv
>
> virtualenv <env-name>
>
> <env-name>\Scripts\activate.bat

## linux:
> source env/bin/activate

## update pip
> python3 -m pip install -U pip

## install requirements
> python3 -m pip install -r requirements.txt

## run flask app
> flask run

## if you want auto reloading upon file change
> python3 app.py


###additional install commands
#####for cuda 10.1:
> pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
> pip install flair