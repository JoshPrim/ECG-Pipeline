 # ECG-Pipeline

### Summary

Medical data such as electrocardiogram recordings have successfully been used as input to artificial intelligence algorithms for the detection of various pathologies. Such algorithms potentially provide non-invasive, relatively low-cost instruments of high diagnostic leverage. However, for supervised learning algorithms such as deep learning models, a considerably large amount of reliable data labelled with correct diagnoses is required. We present a pipeline that processes raw electrocardiogram recordings and prepares them for use as training and validation data for neural network models. Although, the electrocardiogram is a widely used diagnostic instrument, training data appropriately labelled is not only rare but also only available in varying formats from technically differing sources. Therefore, our end-to-end pipeline is designed to flexibly process data from different recording machinery and to read data in PDF format as well as data from native digital devices delivered in XML. We present a use case in which data from XML sources as well as PDF sources is read, cleaned and combined into a unified input dataset for a model predicting myocardial scar as exemplary pathology. The described pipeline will become a cornerstone of our environment for building AI based diagnostic instruments.


### Requirements

Python 3.8

### Support

While we can not provide individual support at the moment, you can see this repository as a public hub to collect feedbacks, bug reports and feature requests.

Please refer to the issue page and feel free to open a ticket.

### Installation

Pull the repository from github.

```
git clone https://github.com/JoshPrim/ECG-Pipeline

cd ECG-Pipeline
```

It is highly reccomended to use a virtual environment for this project, as it requires a specific set of libraries and extensions.
There are multiple ways to set up a virtual environments. The following instructions provide a brief explanation how to set up a virtual environment using Conda. 

Firstly an up-to-date installation of the Anaconda project is needed. A download link can be found here: 
https://docs.anaconda.com/anaconda/install/

After the installation, a virtual environment can be created using the command prompt. 
A simple way of accessing the command prompt in windows requires the press of "Win+R", entering "cmd" in the textfield and pressing the "Enter" key.

The first step to setting up a virtual environment is creating it. This can be done by typing:
```
conda create --name your_env_name python=3.8 -y
```


After creating the environment, additional libraries need to be installed within the environment.
To achieve this the environment needs to be activated by typing:
```
conda activate your_env_name
```

The next step is navigating to this directory via the command prompt (.../ECG-Pipeline) 
a requirements.txt file is provided (TODO: NOTE: ACTUALLY PROVIDE THAT FILE) containing all the required modules.

These modules can be installed by typing: 
```
pip install -r requirements.txt
```
The environment is now ready to be used. 

### Project structure and configration

#### Structure

This pipeline is able to process ECG files from two diffrent ECG manufacturers(Schiller and Cardiosoft) 
In order for the extractors to work properly, the ECG files need to be in their corresponding paths.
A short explanation of the structure of the project follows.

```
ECG-Pipeline
└── data
    ├── kerckhoff
    │   ├── pdf_data
    │   │   ├── pdf_cardiosoft
    │   │   │   ├── clinicalparameters
    │   │   │   ├── extracted_ecgs
    │   │   │   └── original_ecgs
    │   │   └── pdf_schiller
    │   │       ├── clinicalparameters
    │   │       ├── extracted_ecgs
    │   │       └── original_ecgs
    │   └── xml_data
    │       ├── clinicalparameters
    │       └── ecg
    └── metadata

```
If ECG data in XML format are to be used, they must be stored in the 'xml_data' folder. Here the XML files must be placed in the 'ecg' folder and the corresponding clinical parameters in the 'clinicalparameters' folder.

If PDF data are to be used, they must be stored in the 'pdf_data' folder. For PDF data the two providers Cardiosoft and Schiller are currently supported. The untreated PDF data must be placed in the 'original_ecgs' folder.

As this extraction pipeline presents an end to end solution, The corresponding clinical parameters can be inserted into the 'clinicalparameters' folder. This allows classifiers which use the pipeline to have additional information about the patients.
The codes for the clinical parameters and corresponding metadata can be modified in the file in the 'metadata' folder.

#### Configuration

A 'Configuration.ini' file is provided in this project. this file specifies the details for the extraction process. 

To use the extractor, the correct settings need to be entered first. 
This Project also supplies a "Demo" file. To simply run the "Demo" no changes need to be made in the configuration file.

Settings:
* is_pdf = Switch between PDF (True) and XML files (False)
* vis_while_extraction = Switch on(True)/off(False) the visualization during extraction.
* manufacturer = Manufacturer (Possible values: 'Cardiosoft', 'Schiller')



### Execution

For execution of the provided "Demo" a runner is provided. 

There are multiple ways of executing this pipeline, for example, with an IDE or through the command prompt.
An explanation is provided on how to run this pipeline through the command prompt.

To start the runner, the previously created conda environment needs to be active. 
If the environment is still active from the installation, this can be seen in the comand prompt, as it shows the environment name in brackets in front the current directory.
Should the environment not be active, it can be activated again by typing:
```
conda activate your_env_name
```

using the command prompt, navigate to this directory

The demo can be started by typing:
```
python runme.py
```




### Licence 

MIT License (MIT)

