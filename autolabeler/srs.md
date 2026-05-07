Idea:

I am trying to build a labelling tool for preparing datasets. the data i am recording is from hardwares like the spikerbox (brain, neuron, plant or roach) by backyardbrains, BCI devices by nextmind or muse, or any other signal acquisition hardware that collects raw, realtime data using some sort of hardware.

what my lablling tool will help us do is- get necessary input from user like type of hardware, number of classes, what kind of stimulus/movement are we applying or recording and when shall the user apply the stimulus or perform the movement such that it is recorded in the live data. therefore this tool collects information from the user and uses this information to prompt the user to apply appropriate stimulus at appropriate times to prepare a very apt dataset that can be later used to train classification/neural network based models.

in order to get live, raw data from these hardware(sensor) devices, we can design a UDP port which will allow any hardware to stream its raw data on this port and our labelling tool will basically listen for data on this port itself. then get users input for designing the experiment setup and prepare a dataset accordingly. therefore, a user doesn't have to rely on manually starting and stopping recordings, or writing metadata files manually. the tool itself prompts the user to apply touch stimulus(in case of plant spikerbox), or open/close eyes and move right/left arm (in case of BCI devices), or signal the left/right movement(in case of roach spikerbox). and the tool automatically labels the recorded data, records the time when stimulus was applied and stores raw data as dataset.

the final output of this exercise is a model (classification/regression/neural network/etc.) which is trained on this dataset and can be plugged anywhere to get predictions on live data from similar hardware/sensors

---

Below is a **structured Software Requirements & Specification (SRS) document** you can use to present this project to developers, collaborators, or stakeholders. I wrote it in a format similar to what engineering teams and research labs use so it can directly guide development.

---

# Real-Time Biosignal Dataset Labeling Tool

**Software Requirements & Specification (SRS)**

Version: 1.0
Date: March 2026
Author: Siddhesh MM

---

# 1. Project Overview

## 1.1 Purpose

The purpose of this project is to develop a **real-time biosignal dataset labeling tool** that enables researchers and developers to easily record and label signal data collected from electrophysiological and biosensing hardware devices.

The system will allow hardware devices to stream **live raw signal data through a UDP port**, while the software manages experiment timing, stimulus prompts, and automatic labeling of collected signals.

The tool eliminates the need for manual data recording, manual metadata annotation, and manual synchronization between stimulus events and signal recordings.

---

## 1.2 Problem Statement

Most signal acquisition experiments require manual processes such as:

* Manually starting and stopping recordings
* Manually writing metadata files
* Manually synchronizing stimulus events with signal recordings
* Poor dataset structure

These issues lead to:

* Incorrect labeling
* Timing mismatches
* Inconsistent datasets
* Difficulty in training machine learning models

This tool automates the **dataset generation pipeline** from signal acquisition to labeled dataset creation.

---

## 1.3 Goals

The primary goals of the system are:

1. **Collect real-time biosignal data from external hardware**
2. **Synchronize stimulus events with signal recording**
3. **Automatically label datasets**
4. **Provide experiment configuration tools**
5. **Generate structured datasets suitable for ML training**
6. **Support multiple signal acquisition devices**

---

## 1.4 Target Users

Primary users include:

* Neuroscience researchers
* BCI researchers
* Machine learning engineers
* Bioengineering labs
* Students conducting electrophysiology experiments

---

# 2. Scope of the System

The system will support biosignal collection from multiple hardware sources such as:

Examples include:

* Backyard Brains Spikerbox devices (plant, neuron, cockroach)
* Muse EEG headsets
* NextMind BCI devices
* OpenBCI devices
* Other hardware capable of streaming signal data

The system will:

1. Listen for signal data through a **UDP streaming port**
2. Allow users to configure experiments
3. Prompt users to apply stimulus at defined times
4. Automatically record signal segments
5. Generate labeled datasets
6. Store experiment metadata

The system will **not perform model training directly**, but will generate datasets usable by machine learning pipelines.

---

# 3. System Architecture Overview

The system architecture consists of the following components:

```
Hardware Sensors
        │
        ▼
Signal Streaming (UDP)
        │
        ▼
Signal Listener
        │
        ▼
Signal Buffer
        │
        ▼
Experiment Controller
        │
        ▼
Stimulus Scheduler
        │
        ▼
Event Logger
        │
        ▼
Dataset Builder
        │
        ▼
Structured Dataset
```

---

# 4. Functional Requirements

## 4.1 Signal Acquisition

The system must be able to receive real-time signal data from external devices.

Requirements:

* Listen on a configurable UDP port
* Accept continuous signal streams
* Parse incoming signal packets
* Timestamp incoming signals
* Store signals in a real-time buffer

Supported data formats may include:

* CSV formatted packets
* JSON packets
* Binary streams

Example packet format:

```
timestamp,value
1710000000.123,0.34
```

---

## 4.2 Hardware Configuration

Users must be able to define hardware parameters including:

* Hardware type
* Sampling rate
* Number of signal channels
* Signal units

Example configuration:

```
hardware_type: plant_spikerbox
sampling_rate: 1000
channels: 1
```

---

## 4.3 Experiment Configuration

The system must allow users to configure experiment parameters.

Required inputs:

* Number of stimulus classes
* Class labels
* Trial duration
* Rest duration
* Number of trials per class
* Stimulus randomization

Example:

```
Classes:
  - Touch
  - No Touch

Trial duration: 5 seconds
Rest duration: 3 seconds
Trials per class: 30
```

---

## 4.4 Stimulus Prompt System

The tool must guide users during experiments.

Features:

* Countdown before stimulus
* Visual prompts
* Trial progress indicator
* Randomized stimulus order

Example prompts:

```
Trial 1 / 30
Prepare

Stimulus: TOUCH LEAF
Recording...
```

---

## 4.5 Event Logging

The system must record stimulus timing.

Recorded metadata:

* Stimulus start time
* Stimulus end time
* Label name
* Trial number

Example:

```
start_time,end_time,label
3.0,8.0,touch
8.0,11.0,rest
```

---

## 4.6 Dataset Creation

The system must generate structured datasets containing:

* Raw signals
* Event labels
* Experiment metadata

Dataset format example:

```
dataset/
   subject_01/
       session_01/
           signals.csv
           events.csv
           metadata.json
```

---

## 4.7 Real-Time Signal Visualization (Optional)

The system may include live signal visualization.

Features:

* Waveform display
* Signal amplitude
* Time window display

---

# 5. Non-Functional Requirements

## 5.1 Performance

* Signal latency must remain below **50 ms**
* The system must handle sampling rates up to **2000 Hz**
* Packet loss must be tolerated

---

## 5.2 Reliability

* The system must prevent data loss during recording
* All experiment data must be saved automatically

---

## 5.3 Scalability

The system should support:

* Multiple hardware devices
* Multi-channel signals
* Long experiment durations

---

## 5.4 Usability

The system must:

* Provide simple experiment setup
* Provide clear prompts during experiments
* Reduce manual tasks

---

# 6. Data Storage Design

## 6.1 Signal Storage

Signals will be stored as time series data.

Example:

```
timestamp,value
0.001,0.12
0.002,0.13
0.003,0.15
```

---

## 6.2 Event Storage

Stimulus events will be stored separately.

Example:

```
trial,start_time,end_time,label
1,3,8,touch
2,11,16,rest
```

---

## 6.3 Metadata Storage

Experiment metadata will include:

```
hardware_type
sampling_rate
classes
experiment_duration
date
subject_id
```

Stored as:

```
metadata.json
```

---

# 7. System Interfaces

## 7.1 Hardware Interface

Signal data will be received using:

```
UDP socket streaming
```

Configurable parameters:

* IP address
* Port number
* Packet format

---

## 7.2 User Interface

Two possible interfaces:

### CLI Interface

Example command:

```
labeller start experiment_config.yaml
```

### GUI Interface

Features:

* experiment setup panel
* signal visualization
* stimulus prompts

---

# 8. Technology Stack (Proposed)

Backend:

```
Python
FastAPI
asyncio
NumPy
Pandas
```

Signal processing:

```
SciPy
NumPy
```

Visualization:

```
Matplotlib
Plotly
```

Frontend options:

```
PyQt
Electron
React
```

---

# 9. Future Extensions

Potential improvements include:

* Real-time machine learning inference
* Hardware-specific adapters
* Dataset export to ML frameworks
* Support for Lab Streaming Layer (LSL)
* Cloud dataset storage

---

# 10. Risks and Challenges

Potential technical challenges include:

* Signal timing synchronization
* Hardware compatibility
* UDP packet loss
* High sampling rate handling
* Noise filtering

---

# 11. Expected Output

The final system should produce datasets that can directly be used for machine learning tasks such as:

* Signal classification
* Movement detection
* Stimulus detection
* Brain-computer interface models

Example ML workflow:

```
Dataset → Feature Extraction → Model Training → Real-time Prediction
```

---

If you'd like, I can also help you create a **much stronger version of this document that includes:**

* System architecture diagrams
* Data flow diagrams
* API specifications
* UDP packet protocol specification
* Plugin system for different hardware devices

That would turn this into a **proper engineering design document used by real dev teams.**
