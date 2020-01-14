# universal-vehicle-analysis-paper
This repository contains results for paper submission for The International Conference on Computational Science 2020 [(IССS website)](https://www.iccs-meeting.org/iccs2020/)

The paper was submited on track 18: Smart Systems: Bringing Together Computer Vision, Sensor Networks and Machine Learning – SmartSys [(link)](https://www.iccs-meeting.org/iccs2020/thematic-tracks/#ws9)


### Paper Abstract:
This article considers the vehicle analysis problem. In order to solve this, we propose a system that recognises the parameters of the vehicle, such as the licence plate, brand, colour and direction, and tracks it over time. This kind of analytical approach has many potential applications, ranging from security to traffic control. We propose an integrated approach to solving the problem that uses an intelligent recognition system with an emphasis on licence plate recognition, and which is based on the use of deep learning algorithms, including convolutional neural networks. Our approach is resistant to variations in licence plate templates in different countries and continents, camera locations and the quality of input images, as well as to changes in lighting, weather conditions, deformation and licence plate features. The quality of the system is ensured by the optimisation of various models. The convolutional neural networks were trained using images from several datasets, and data augmentation techniques were applied to achieve stability under various conditions. To verify the effectiveness of our solution, we present the results of numerical experiments that confirm the superiority and comparability of detecting and recognising licence plates using the 2.7.102 cloud version of the commercial OpenALPR package in public datasets. For the 2017-IWT4S-HDR_LP public dataset, the accuracy of licence plate recognition was 94%, and for the Application-Oriented Licence Plate public dataset, this was 88%. The resulting algorithm is resistant to changes in the size of the input image, features and rules for constructing licence plates, and its results can be localised to any country. A current system demo is available at [proposed system demo](https://broutonlab.com/solutions/plate-number-recognition)


### Results of dataset evaluation and comparison
We test our solution with OpenALPR Cloud API [link](https://api.openalpr.com/v2/)

links to results, described at paper
+ AOLP dataset: 

[link to download page](http://aolpr.ntust.edu.tw/lab)
 
 
[results provided in paper (google drive link)](https://drive.google.com/file/d/1mN9pCk3f9hi73Py4XTip7-kSEzurgHg5/view?usp=sharing)
+ 2017-IWT4S-HDR_LP-dataset: 

[link to download page](https://medusa.fit.vutbr.cz/traffic/research-topics/general-traffic-analysis/holistic-recognition-of-low-quality-license-plates-by-cnn-using-track-annotated-data-iwt4s-avss-2017/) 

[results provided in paper (google drive link)](https://drive.google.com/open?id=11ZgIFvhLbJ6gb1iKo8Vyg6ZThrIklopi)