# XRP Primer AI Kit Demo

## Project Description

This repository provides a guide for setting up a local Large Language Model (LLM) serving environment on a Rubik Pi 3 (RPi3), utilizing Ubuntu Server, Python 3.12, and the Ollama framework.

## 1. Initial Hardware Setup: Ubuntu OS on the Rubik Pi 3

The first step is to follow [this guide](https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.0.0-u/Update-Software/3.2.Flash-using-Qualcomm-Launcher) to install Ubuntu Server on the Rubik Pi 3. Then follow [these instructions](https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.0.0-u/Troubleshooting/11.3.ubuntu-desktop-vs-server/#switch-from-ubuntu-desktop-version-to-server-version) to switch to ubuntu desktop.

**Abridged OS install instructions**

Please read the official documentation linked above very closely. That said, here is summary of the steps required
- Put the Rubik Pi in EDL mode
- Flash Ubuntu with Qualcomm-Launcher
	- default username: ubuntu
	- default password: ubuntu
	- you will be asked to change the default password
	- reccomended new password: xrpaikit
	- you will be asked to setup a connection to a local wifi network (important)
- Use your host machine to connect to the Rubik Pi via UART with Putty
	- find COM in Device manager, Baud is 115200
	- run the following commands
		- sudo apt install qcom-adreno1- libgbm-msm1- libegl-mesa0 libegl1 libgles2 libglvnd0 libvulkan1 weston-
		- sudo apt update --fix-missing
		- sudo apt install ubuntu-desktop
		- sudo apt update --fix-missing
		- sudo reboot
- You can now plug HDMI/mouse/keyboard into the Rubik Pi and access the desktop environment

## 2. Setting Up the Development Environment

After logging in, execute the following commands to install necessary dependencies and clone the project.

**System Update:**
```bash
sudo apt update
sudo apt upgrade -y
```

**Install Git:**
```bash
sudo apt install git -y
```
**(Best practice) create a projects folder in your **
```
mkdir -p projects
cd projects
```

**Clone the Project:**
```bash
git clone https://github.com/SaintSampo/Primer-Software
cd Primer-Software
```

### 2.2. Installing Python 3.12

**Install Python 3.12 and Venv Module:**
```bash
sudo apt install python3.12 python3.12-venv -y
```

**Verify Installation:**
```bash
python3.12 --version
```

**Essential Installs:**
```bash
sudo apt install -y build-essentials python3-dev libportaudio2 portaudio19-dev
```

### 2.3. Installing Ollama

Ollama is the core LLM serving application. Its installer script automatically handles architecture detection for the ARM platform.

**Execute the Ollama Installer:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify Service Status:**
Check that the Ollama service is running in the background.
```bash
systemctl status ollama
```

**Run a Test Model:**
Download a lightweight model to confirm functionality. (Due to RPi3's 1GB RAM, always choose small, highly quantized models like `tinyllama`).
```bash
ollama run tinyllama
```
Type `/bye` to exit the interactive session.

## 2.4 Whisper Model Setup

**1. Download the Models:**
You must manually download the following files and place them in the specified directory:
- **Decoder Model:** Download `decoder_model.onnx` from [Here](https://huggingface.co/onnx-community/whisper-tiny.en/blob/main/onnx/decoder_model.onnx)
- **Encoder Model:** Download `encoder_model.onnx` from [Here](https://huggingface.co/onnx-community/whisper-tiny.en/blob/main/onnx/encoder_model.onnx)

**2. Placement:**
Place both files into the following path within your cloned repository:
`lib/whisper/`

## 3. Python Development Environment

### 3.0 install system requirement library

```bash
sudo usermod -a -G gpio ubuntu
```

### 3.1. Creating a Virtual Environment (venv)

Isolating dependencies in a virtual environment is essential for project stability.

**Create Venv using Python 3.12:**
```bash
python3.12 -m venv venv
```

**Activate the Virtual Environment:**
```bash
source venv/bin/activate
```
> **Note**: Your command prompt will change (e.g., `(venv)user@host:~...`) indicating the environment is active. You must run this command every time you open a new terminal session.

### 3.2. Installing Dependencies

Install all necessary libraries defined in the `requirements.txt` file.

**Install Packages:**
```bash
pip install -r requirements.txt
```

The Rubik Pi 3 is now fully configured.

### 3.3. Run the Demo

Now that the virtual enviornment is active with all the needed dependancies, you can test speech to text by running:
```bash
./venv/bin/python src/whisper_prompt.py
```
or the language model with:
```bash
./venv/bin/python src/ollama_chat.py
```
if you want to test the display, you can with:
```bash
sudo ./venv/bin/python src/render_avatar.py
```
