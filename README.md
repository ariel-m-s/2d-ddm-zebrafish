# Useful things to install

Note that for some of these things to work, it's likely that you'll need to restart your command line (terminal) after their installation.

## Package managers

A package manager is a tool that automates the process of installing, upgrading, configuring, and removing software packages on your computer. It is a very useful tool, but it is not strictly necessary.

### Homebrew for Mac and Linux

Open your terminal and run the following command:

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

To check if `brew` was installed correctly, run the following command:

```shell
brew --version
```

For more information on the Homebrew package manager, visit the [Homebrew website](https://brew.sh/).

### Chocolatey for Windows

Open your PowerShell **as administrator**.

Then, run the following command (you can also install it as non-administrator, but it's a bit more complicated):

```shell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Depending on your computer's settings, you might need to ensure that the execution policy is not set to `Restricted`. If not, the installation will fail with a message that says something like "execution of scripts is disabled on this system". You can check this by running the following command:

```shell
Get-ExecutionPolicy
```

If the output is `Restricted`, it means that you must change the execution policy as follows:

```shell
Set-ExecutionPolicy AllSigned
```

or

```shell
Set-ExecutionPolicy RemoteSigned
```

Once the installation is complete, you can change the execution policy back to `Restricted` to keep your computer safe:

```shell
Set-ExecutionPolicy Restricted
```

To check if `choco` was installed correctly, run the following command:

```shell
choco --version
```

For more information on the Chocolatey package manager, visit the [Chocolatey website](https://chocolatey.org/install), and for more information on the execution policy, visit this [StackOverflow post](https://stackoverflow.com/questions/4037939/powershell-says-execution-of-scripts-is-disabled-on-this-system).

## Make

In software developement, it is common to use a tool called `make` to automate tasks. It is a very powerful tool, but in this project it is just used to create shortcuts for the most common tasks by simply running the `make <shortcut>` commands. Because of this, `make` is not strictly necessary, but it is quite useful. If not installed, you'll have to go into the [`Makefile`](https://gitlab.ethz.ch/ibt/recon/projects/ztesaropt/-/blob/main/Makefile) to copy and paste the commands into your terminal.

### Installation on Mac and Linux (with Homebrew)

Open your terminal and run the following command:

```shell
brew install make
```

Or, to install `make` together with many other useful software development tools for Mac, run the following command (just for Mac):

```shell
xcode-select --install
```

For more information, visit this [StackOverflow post](https://stackoverflow.com/questions/10265742/how-to-install-make-and-gcc-on-a-mac).

### Installation on Windows (with Chocolatey)

Open your PowerShell **as administrator** and run the following command:

```shell
choco install make
```

For more information, visit this [StackOverflow post](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows).

Note that you might need to change the execution policy again (as explained before) also to run the `make` commands.

## Python 3

Python is a programming language used in this project. It is very popular, and there are many libraries available for it.

If you don't have Python 3 installed, please download it directly from the [Python website](https://www.python.org/downloads/). This program works with any Python version between 3.8.x and 3.11.x. It should work with any Python 3 version greater or equal to 3.8.0 but, by the time of this writing, the latest version of Python is 3.11.2.

After having cloned the repository (see below), make sure that the variables `PYTHON` and `PIP` in the [`Makefile`](https://gitlab.ethz.ch/ibt/recon/projects/ztesaropt/-/blob/main/Makefile) point to the correct executables. If you are using Windows, set `PYTHON = py` (or `PYTHON = python` if you installed Python 3 through the Microsoft Store) and `PIP = pip`. If you are using Mac or Linux, set `PYTHON = python3` and `PIP = pip3` (this is less ambiguous because the other ones can point to Python 2).

## Poetry

Poetry is a tool for dependency management and packaging in Python. It is used to create virtual environments, install dependencies, and more. It is very useful, but it is not strictly necessary. If you don't want to use Poetry, you can install the required packages (listed in the `pyproject.toml` file) manually or using another dependency manager (e.g. `pip`, `conda`, `pipenv`). However, I will only provide instructions for installing the packages with Poetry.

For detailed installation instructions refer to the [Poetry website](https://python-poetry.org/docs/).

### What is a virtual environment?

A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated Python environments for them. It solves the "Project X depends on version 1.x but Project Y needs 4.x" dilemma, and keeps your global site-packages directory clean and manageable. For more information, visit the [Python Virtual Environments documentation](https://docs.python.org/3/tutorial/venv.html).

### Installation on Mac and Linux

Open your terminal and run the following command:

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

### Installation on Windows

Open your PowerShell **as administrator** and run the following command:

```shell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

If you have installed Python through the Microsoft Store, replace `py` with `python` in the command above, as follows:

```shell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Pay attention to the output of the command, as it will tell the path where Poetry was installed. In my case, it was `C:\Users\<user>\AppData\Roaming\Python\Scripts`, where `<user>` is my username. If you are using Windows, after having cloned the repository (see below), make sure that the variable `POETRY` in the [`Makefile`](https://gitlab.ethz.ch/ibt/recon/projects/ztesaropt/-/blob/main/Makefile) points to the correct executable. In my case, it was:

```
POETRY = C:\Users\<user>\AppData\Roaming\Python\Scripts\poetry
```

Note that I added `\poetry` to the end of the path. That is the Poetry executable.

## Notes for specific OSs and architectures

I don't know the specifications for every computer, so contact me at martinea\[at\]student\[dot\]ethz\[dot\]ch in case of bugs.

### Note for Apple silicon users (MX)

Make sure you have Rosetta 2 on your computer, by running the following command:

```shell
softwareupdate --install-rosetta
```

Then install `hdf5` with Homebrew:

```shell
brew install hdf5
```

### Note for Windows users

All Windows installation instructions were tested on **Windows 11** and using the administrator mode of the PowerShell. An alternative way to install the requirerements for Windows is to use the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) (WSL) and follow the instructions for Linux.

---

# Contributing to the project

Before writing any code, make sure that you've created a development-friendly virtual environment:

```shell
make createvenv
```

To add new Python dependencies (libraries), run the following Poetry command:

```shell
<poetry> add <dep>
```

where `<poetry>` is the path to the Poetry executable (as explained before) and `<dev>` is the name of the library. For example, if I want to install `numpy` (don't do this, as it is already installed), I would run:

```shell
poetry add numpy
```

If you added this dependency locally, that's all you have to do. You can continue writing code and running the program as always.

You can also update the program using version control (_e.g._, `git pull`). If no new dependencies have been added, no further actions need to be taken. However, if a new dependency was added, you will have to update the virtual environment. You can do this with the same command as before:

```shell
make createvenv
```
