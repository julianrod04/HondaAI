# Google Cloud VM and CARLA setup

## Table of Contents
1. Introduction
2. VM Configuration & SSH Connection
3. Installing Chrome Remote Desktop (CRD)
4. Installing NVIDIA Drivers
5. Installing and Running CARLA

## 1. Introduction

This is a guide on how to setup a Google Cloud Virtual Machine (VM) and how to install/run CARLA. This guide does NOT cover how to setup a Google Cloud account. Payment may be required.

As of October 2025, Google Cloud is offering a 90-day, $300 free trial. Learn more here: https://cloud.google.com/free/docs/free-cloud-features.

After setting up a Google Cloud account and acquiring credits, begin by creating a compute VM instance.

1. Start from the Google Cloud Console (homepage)
2. Go to the navigation menu
3. Under "Compute Engine", click on "VM Instances"
4. Click on "Create instance" to start the VM configuration

## 2. VM Configuration & SSH Connection

### VM Configuration

Configure the VM. Any configuration not listed will be left in the default setting.

1. Machine Configuration
    * Name: ```seniordesign-heavy```
    * Region: ```us-east1 (South Carolina)```
    * Machine Type: ```g2-standard-32 (32 vCPU, 16 core, 128GB memory) [GPU Preset]``` *

2. OS and Storage
    * OS: ```Ubuntu```
    * Version: ```Ubuntu 22.04 LTS for x86``` **
    * Disk Size: ```256GB``` ***

3. Networking
    * ```Allow HTTP```
    * ```Allow HTTPS```

4. Security
    * ```Allow full access to all Cloud APIs```

### Connecting to the VM via SSH

In the VM Instances menu, connect to SSH via "SSH" under "Connect". Authorize SSH access via browser.
\
\
\
NOTES

A **cloud quota** is the limit set on how much of a cloud resource you’re allowed to use (like storage, CPU, or bandwidth).

Before requesting quota increase, first select the desired configuration. Quota should be requested when prompted to take action on a quota issue. Per previous experience, exact quota request is recommended instead of estimated future quota needs. Requesting excess resources will likely be result in a declined request (e.g. requesting 4 GPUs when only 2 are needed).

\* This guide uses 32 vCPU, which is more than enough compute. An 8 vCPU, 32GB RAM instance should suffice.

\*\* LTS = Long-term support

\*\*\* CARLA 20GB + UnReal Engine 130GB + Misc.

## 3. Installing Chrome Remote Desktop (CRD)

The following instructions are based on the official Google Cloud guide (https://cloud.google.com/architecture/chrome-desktop-remote-on-compute-engine)

### Install CRD and install a desktop environment. 
This setup will install the lightweight ```Xcfe``` desktop environment. Read more about Xcfe here: https://www.xfce.org/

```
curl https://dl.google.com/linux/linux_signing_key.pub \
    | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/chrome-remote-desktop.gpg
echo "deb [arch=amd64] https://dl.google.com/linux/chrome-remote-desktop/deb stable main" \
    | sudo tee /etc/apt/sources.list.d/chrome-remote-desktop.list
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive \
    apt-get install --assume-yes chrome-remote-desktop

sudo DEBIAN_FRONTEND=noninteractive \
    apt install --assume-yes xfce4 desktop-base dbus-x11 xscreensaver
sudo bash -c 'echo "exec /etc/X11/Xsession /usr/bin/xfce4-session" > /etc/chrome-remote-desktop-session'
sudo systemctl disable lightdm.service
sudo apt install --assume-yes task-xfce-desktop
```

**Note**: The command ```sudo systemctl disable lightdm.service``` may return an error, like ```Failed to disable unit: Unit file lightdm.service does not exist.```. This is because [explain]. Ignore the error.

### Configure and start CRD. 

Go to https://remotedesktop.google.com/headless.
1. Click Begin > Next > Authorize.
2. Copy the command under "Debian Linux". It should look like the command below.
```
DISPLAY= /opt/google/chrome-remote-desktop/start-host \
    --code="4/xxxxxxxxxxxxxxxxxxxxxxxx" \
    --redirect-url="https://remotedesktop.google.com/_/oauthredirect" \
    --name=$(hostname)
```
3. Run the command in the VM
4. Create a pin for CRD. *Do not lose this pin*.
5. Verify with ```sudo systemctl status chrome-remote-desktop@$USER```. The output should include ```Active: active (running)```. Use ```Ctrl + C``` to exit.
6. Go to ```https://remotedesktop.google.com/access``` and refresh. The VM is now accessible via CRD.

### Optional
Optionally install Chrome (really, it's optional)
```
curl -L -o google-chrome-stable_current_amd64.deb \
https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install --assume-yes --fix-broken ./google-chrome-stable_current_amd64.deb
```

## 4. Installing NVIDIA Drivers

NVIDIA drivers are not installed by default. They must be installed before using any NVIDIA GPU.

This guide uses the ```ubuntu-drivers-common``` tool to automatically install required drivers.

```
sudo apt install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall
sudo reboot
```

The above commands will reboot the VM. After reconnecting, the command ```nvidia-smi``` should now show GPU details.

## 5. Installing and Running CARLA

### Install CARLA and required tools
```
mkdir carla_simulator
cd carla_simulator
wget -O CARLA_0.9.16.tar.gz https://tiny.carla.org/carla-0-9-16-linux
wget -O AdditionalMaps_0.9.16.tar.gz https://tiny.carla.org/additional-maps-0-9-16-linux
tar -xvf CARLA_0.9.16.tar.gz
tar -xvf AdditionalMaps_0.9.16.tar.gz
./ImportAssets.sh

sudo apt install python3-pip -y
sudo apt install python3.12-venv -y
python3 -m venv .carla_venv
source .carla_venv/bin/activate

python3 -m pip install numpy pygame carla

sed -i 's/^bUseMouseForTouch=False$/bUseMouseForTouch=True/' CarlaUE4/Config/DefaultInput.ini

```

Note: CARLA does not correctly handle mouse free-look by default, which is why ```bUseMouseForTouch``` is updated.

### Start the CARLA server and player

Beyond this point, commands should be run in the VM Desktop 

**For each terminal, ensure current directory is CARLA root folder (~/carla_simulator) and activate the virtual environment (```source .carla_venv/bin/activate```)**

**Terminal Tab 1**

```
./CarlaUE4.sh
```

This will start the server and open the server-view window.

**Terminal Tab 2**

```
./PythonAPI/util/config.py --map Town06
python3 PythonAPI/examples/manual_control.py
```

This will change the server-view to map Town06, and open the user-controlled car game mode.

### Notes

CARLA Installation and Setup: https://www.youtube.com/watch?v=tV6iO8JikTw

Server mouse sentitivity issue: https://github.com/carla-simulator/carla/issues/3579