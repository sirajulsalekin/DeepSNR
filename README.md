# DeepSNR
Base-pair resolution detection of transcription factor binding site by deep deconvolutional network

Authors: Sirajul Salekin, Jianqiu (Michelle) Zhang and Yufei Huang

ABSTRACT

Transcription factor (TF) binds to the promoter region of a gene to control gene expression. Identifying precise transcription factor binding sites (TFBS) is essential for understanding the detailed mechanisms of TF mediated gene regulation. However, there is a shortage of computational approach that can deliver single base pair (bp) resolution prediction of TFBS. In this paper, we propose DeepSNR, a Deep Learning algorithm for predicting transcription factor binding location at Single Nucleotide Resolution. DeepSNR adopts a novel deconvolutional network (deconvNet) model and is inspired by the similarity to image segmentation by deconvNet. The proposed deconvNet architecture is constructed on top of 'DeepBind' and we trained the entire model using TF specific data from ChIP-exonuclease (ChIP-exo) experiments. DeepSNR has been shown to outperform motif search based methods for several evaluation metrics. We have also demonstrated the usefulness of DeepSNR in the regulatory analysis of TFBS as well as in improving the TFBS prediction specificity using ChIP-seq data.

--------------------------------------------------------------------------------------------

After cloning or downloading the repository, you need to run the "DeepSNR_main.py" script to get the prediction of CTCF transcription factor.

For other TFs, the model definition need to revised according to the information provided in the "DeepSNR model_ .txt" file of the corresponding TF directory.

