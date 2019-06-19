# IEEE ICMLA 2019 Challenge
## Protein Inter-Residue Distance Prediction
https://www.icmla-conference.org/icmla19/challenges.html

## Organizers  
[Badri Adhikari](https://badriadhikari.github.io/) and [Sharlee Climer](http://www.cs.umsl.edu/~climer/)  
University of Missouri-St. Louis

## Contact  
adhikarib@umsl.edu

## Important Dates  
Paper Submission Deadline: September 14, 2019  
Notification of acceptance: : October 7, 2018  
Camera-ready papers & Pre-registration: October 17, 2018  
The IEEE ICMLA Conference: December 16-19, 2019  

##  Datasets & Code
* The dataset for training, validation, and testing are available at [Google Cloud Storage](https://console.cloud.google.com/storage/browser/protein-distance) 
* A Python Notebook containing code to train, validate, and test is available at this repository - 'PIDP_v1_0.ipynb'. GitHub sometimes fails to open this notebook (probably because of its large size). [Here](https://colab.research.google.com/drive/1w4_ETou_y5455VQD2bbF_4R6bI2GQqWo) is the Google Colaboratory link to the notebook.

##  Overview  
Predicting three-dimensional structures of proteins is a notoriously challenging interdisciplinary problem [1,2]. Many biologists, biochemists, bioinformaticians, biomedical researchers, and computer scientists have been attacking the problem for 50+ years . Now, with the help of deep learning methods and high throughput protein sequencing technologies, it appears that we are close to cracking it. The overall goal in the field of protein structure prediction is to predict full three-dimensional structure given an amino acid sequence. Google DeepMind recently participated in the CASP13 protein structure prediction challenge as the AlphaFold group  and contributed some novel ideas to the field ([read more](https://moalquraishi.wordpress.com/2018/12/09/alphafold-casp13-what-just-happened/)). It is clear that there are many unexplored opportunities in the field using the data that is currently available.  

In this challenge, we focus on predicting the inter-residue distances (in Angstroms) between the residues (carbon-beta atoms) in a protein. For a protein sequence, we first predict useful features such as secondary structures and coevolutionary signals, which serve as the input to a neural network. The network, in turn, should aim to predict the physical distances between all pairs of amino acids.  
<p align="center">
<img src="pidp-overview.png" align="middle" height="450"/></br>
Figure1. A correct distance matrix is obtained from an experimentally determined structure of a protein (left). The goal is to predict these distances using deep learning with the help of features predicted from the corresponding sequence (right).
</p>
From the perspective of input and output data format, the protein distance prediction problem is similar to depth prediction [3] in computer vision, i.e. predicting 3D depth from 2D images. In the depth prediction problem, the input is an image of dimensions H x W x C, where H is height, W is width, and C is number of input channels, and the output is a two-dimensional matrix of size H x W whose values represent the depth intensities. Similarly, in the protein distance prediction problem, the input is protein features of dimension L x L x N, where L is the length of the protein sequence, N is the number of input channels, and the output is a distance matrix of size L x L. Depth prediction usually involves three channels (red, green, and blue or hue, saturation, and value) while in the latter we have much higher number of features such as 56 [4] or 441 [5]. The dataset in this challenge, however, has only 13 input channels. These input channels represent features such as secondary structures, solvent accessibility, and coevolutionary signals predicted using tools such as CCMpred [6] and FreeContact [7].

## Challenge  
Each protein input features are an input volume of size 256 x 256 x 13 and the output labels are a 256 x 256 matrix of real numbers that represent the physical distances (from 3 Angstroms to 100+ Angstroms). The challenge here is to learn from the 13 input channels to accurately predict the output labels. It is a regression problem where we predict real values. However, it can be transformed into a multi-class classification problem by creating bins of distance ranges, i.e. predict whether each distance maps to a range of 0 to 3, or 3 to 6, or 6+, etc. It is expected that you train and validate your method on the Training Dataset and finally test it on the Test Dataset.

## Datasets and Codes  

The dataset for training, validation, and testing are available at the Google Storage location. 

A Python Notebook with code to train, validate, and test the performance is available at the GitHub repository. 

### Training & Validation Dataset  

The training and validation dataset consists of 3,420 proteins. This dataset is available as two numpy files: 1) full-distance-maps-cb.npy (800 MB), and 2) full-input-features.npy (1.9 GB). Samples of these two files (200 proteins only) are also available for download. These files are 50 MB and 115 MB and are ideal for prototyping.

### Test Dataset  

The test dataset consists of 150 proteins. These are available as two numpy files: 1) testset-distance-maps-cb.npy, and 2) testset-input-features.npy.

### Codes  

The python notebook file PIDP_v1_0.ipynb has the code to: 1) load the dataset (sample dataset by default), 2) build a small ConvNet (an example model), 3) train and validate the model, and 4) evaluate the model on the Test dataset. 

### Building 3D Models using the predicted distances

The folder “how_to_build_models” in the this GitHub repository contains the code examples to build three dimensional models and to evaluate the models against the true/correct structures.

## How Did We Prepare the Dataset?

We prepared input features using the following six methods: 1) Psipred, 2) Psisolv, 3) Shannon entropy sum, 4) Ccmpred, 5) Freecontact, and 6) Pstat_pots. Of these six, the first three are 1D features, i.e. the lenth of the feature is same as the protein sequence length (L), and the last three are 2D features, i.e. their length is L * L. Psipred, in particular, has three rows of inputs. When all these 1D features are converted to 2D, Psipred features end up as 6 2D features (6 channels), Psisolv as 2 2D features, and Shanon entrohy sum as 2 2D features. The last three features are each 1 2D channel. Hence, in total we have 13 input channels (6 + 2 + 2 + 1 + 1 + 1 = 13). Please check the Perl script file "prepare-features.pl" in this repository for the details.

## Performance Evaluation  
The goal of our evaluation is to assess the usefulness of predicted distances towards using them to build full three-dimensional models. Long-range distances are the most difficult to accurately determine and our evaluations focus on ‘long-range’ distances, as shown in Figure 2. Long-range distances in a distance matrix are the distances between pairs of amino acids that are more than 23 residues apart in the corresponding protein sequence. Towards that goal, we will evaluate distance map predictions in two ways: 

<b>1. MAE of the top L long-range smallest predicted distances:</b> Of all the predicted long-range distances, distances are sorted in ascending order and only the top L distances are considered for evaluation. L stands for the length of the protein being evaluated. Then the mean absolute error (MAE) between these predicted distances and true distances is calculated.  

<b>2. Precision & Recall of all long-range contacts:</b> From the true distance matrix and the predicted distance matrix, all long-range distances less than 8 Angstroms are set to 1 and all others to zero, i.e. we convert the distance matrix to long-range contact matrix. Then, we calculate precision as the ratio of the number of matches and the total number of long-range contacts in the true contact matrix.  
<p align="center">
<img src="what-is-long-range.png" align="middle" height="250"/></br>
Figure2. Long-range distances and non-long-range distances in a distance matrix.      
</p>
<b>3. TMscore and RMSD of the top one model:</b> The distance maps predicted can be used to build full three-dimensional models (see the folder “how_to_build_models” in the GitHub repository). The quality of the three-dimensional models will be evaluated using template-modeling score (TMscore) and root mean square deviation (RMSD) calculated using the TMscore tool at https://zhanglab.ccmb.med.umich.edu/TM-score/.

## Submission

A short paper (4 pages) describing the proposed algorithms and results on the provided datasets should be submitted through the main conference submission website. We also require you to submit your code along with your paper, although you may choose to not to release your code publicly. Submitted papers will be reviewed mainly based on: 1) originality and technical soundness of the employed algorithm, and 2) performance of the algorithm with respect to the evaluation criterion above. Submit your paper at https://www.icmla-conference.org/icmla19/howtosubmit.html.

## Publication

Accepted papers will be scheduled for presentations at the conference and published in the IEEE ICMLA 2019 conference proceedings.

## Where to Start?

We suggest that you start with the Python Notebook in the GitHub repository. At Google Colaboratory (https://colab.research.google.com) you can create an empty notebook and test the code. All you need for this is a Google account. The notebook code loads the “sample” dataset. This can be a free and convenient platform for testing your ideas.

You (probably) won’t be able to load the full dataset in Google Colaboratory. You will need 8 to 30 GB of RAM and a good GPU (such as Tesla K80) to perform the full training and validation. If you do not have such hardware resources, we encourage you to apply to Google’s “GCP research credits program” at https://lp.google-mkto.com/gcp-research-credits-FAQ.html. The GitHub repository has a folder that has instructions and sample code that demonstrates how jobs can be submitted to GCP’s ML Engine.

## References

1. 	Adhikari B. DEEPCON: Protein Contact Prediction using Dilated Convolutional Neural Networks with Dropout. bioRxiv. 2019; 590455. doi:10.1101/590455
2. 	Wang S, Sun S, Li Z, Zhang R, Xu J. Accurate De Novo Prediction of Protein Contact Map by Ultra-Deep Learning Model. PLoS Comput Biol. 2017; doi:10.1371/journal.pcbi.1005324
3. 	Eigen D, Puhrsch C, Fergus R. Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. 
4. 	Adhikari B, Hou J, Cheng J. DNCON2: improved protein contact prediction using two-level deep convolutional neural networks. Bioinformatics. 2017; doi:10.1093/bioinformatics/btx781
5. 	Jones DT, Kandathil SM. High precision in protein contact prediction using fully convolutional neural networks and minimal sequence features. Valencia A, editor. Bioinformatics. 2018; doi:10.1093/bioinformatics/bty341
6. 	Seemayer S, Gruber M, Söding J. CCMpred - Fast and precise prediction of protein residue-residue contacts from correlated mutations. Bioinformatics. 2014;30: 3128–3130. doi:10.1093/bioinformatics/btu500
7. 	Kaján L, Hopf T a, Kalaš M, Marks DS, Rost B. FreeContact: fast and free software for protein contact prediction from residue co-evolution. BMC Bioinformatics. 2014;15: 85. doi:10.1186/1471-2105-15-85
8. 	Adhikari B, Bhattacharya D, Cao R, Cheng J. CONFOLD: Residue-residue contact-guided ab initio protein folding. Proteins. 2015;83: 1436–49. doi:10.1002/prot.24829
9. 	Adhikari B, Nowotny J, Bhattacharya D, Hou J, Cheng J. ConEVA: a toolbox for comprehensive assessment of protein contacts. BMC Bioinformatics. 2016;17: 517. doi:10.1186/s12859-016-1404-z
