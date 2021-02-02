 # ECG-Pipeline

### Summary

Medical data such as electrocardiogram recordings have successfully been used as input to artificial intelligence algorithms for the detection of various pathologies. Such algorithms potentially provide non-invasive, relatively low-cost instruments of high diagnostic leverage. However, for supervised learning algorithms such as deep learning models, a considerably large amount of reliable data labelled with correct diagnoses is required. We present a pipeline that processes raw electrocardiogram recordings and prepares them for use as training and validation data for neural network models. Although, the electrocardiogram is a widely used diagnostic instrument, training data appropriately labelled is not only rare but also only available in varying formats from technically differing sources. Therefore, our end-to-end pipeline is designed to flexibly process data from different recording machinery and to read data in PDF format as well as data from native digital devices delivered in XML. We present a use case in which data from XML sources as well as PDF sources is read, cleaned and combined into a unified input dataset for a model predicting myocardial scar as exemplary pathology. The described pipeline will become a cornerstone of our environment for building AI based diagnostic instruments.


### Project structure

Todo ...

### Requirements

Python 3.5+

### Support

While we can not provide individual support at the moment, you can see this repository as a public hub to collect feedbacks, bug reports and feature requests.

Please refer to the issue page and feel free to open a ticket.

### Installation & Usage

Pull the repo from github.

```
git clone https://github.com/JoshPrim/ECG-Pipeline

cd EVA-Projekt
```

Todo ...




### Licence 

MIT License (MIT)

