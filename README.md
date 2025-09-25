# Resilient Source Seeking with robot swarms - 2D distributed 

This repository contains some Python simulations that help us to validate the results and findings presented in the following article:
```
@misc{jbautista2025distributedss,
  title={Fully distributed and resilient source seeking for robot swarms}, 
  author={Jesus Bautista and Antonio Acuaviva and Jose Hinojosa and Weijia Yao and Juan Jimenez and Hector Garcia de Marina},
  year={2025},
  url={},
}
```
    
## Installation

We recommend creating a dedicated virtual environment to ensure that the project dependencies do not conflict with other Python packages:
```bash
python -m venv venv
source venv/bin/activate
```
Then, install the required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ```requirements.txt``` contains the versions tested for **compatibility with the simulator**.
Do **not modify the versions** to ensure stable and reproducible environments. Note that ```ssl_simulator``` already provides stable versions for the following core packages: ```numpy```, ```matplotlib```, ```tqdm```, ```pandas```, ```scipy```, ```ipython```.

### Additional Dependencies
Some additional dependencies, such as LaTeX fonts and FFmpeg, may be required. We recommend following the installation instructions provided in the ```ssl_simulator``` [README](https://github.com/Swarm-Systems-Lab/ssl_simulator/blob/master/README.md). 

To verify that all additional dependencies are correctly installed on Linux, run:
```bash
bash test/test_dep.sh
```

## Usage

Run the Jupyter notebooks inside the `notebooks` directory and/or the Python application inside `apps`.

## Credits

If you have any questions, open an issue or reach out to the maintainers:

- **[Jesús Bautista Villar](https://sites.google.com/view/jbautista-research)** (<jesbauti20@gmail.com>) – Main Developer
