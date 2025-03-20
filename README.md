# MedicalVLM-matching

## Survey results for paper named "Do medical VLMs discover discriminative features in multi-modal medical images?"
### Dataset about image classification across various modalities
|dataset name|dataset portal|task|modality|DOI of dataset|remarks|
|:-|:-|:-|:-|:-|:-|
|Brain tumor dataset|[figshare](https://doi.org/10.6084/m9.figshare.1512427.v8)|classification<br>+detection|MRI|10.1371/journal.pone.0140381|This dataset includes images on three imaging plane: axial, coronal, sagital.|
|Brain tumor classification|[kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)|classification|MRI|10.34740/kaggle/dsv/1183165|
|CheXpert|[HP](https://stanfordmlgroup.github.io/competitions/chexpert/), [kaggle](https://www.kaggle.com/datasets/ashery/chexpert)|classification|X-ray|10.1609/aaai.v33i01.3301590|
|NIH Chest X-ray Dataset|[Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)|classification|X-ray||
|LIDC-IDRI|[TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/)|classification<br>+detection|CT|10.7937/K9/TCIA.2015.LO9QL9SX|
|COVIDx CT|[kaggle](https://www.kaggle.com/datasets/hgunraj/covidxct)|classification|
|Breast Ultrasound Images Dataset|[kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)|classification|US|10.1016/j.dib.2019.104863|
|DiagSet|https://ai-econsilio.diag.pl/|classification|histpathology|10.1038/s41598-024-52183-4|
|BreakHis|[Their HP](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)|classification|histpathology|10.1109/TBME.2015.2496264|
|HiCervix|[Zenodo](https://zenodo.org/records/11087263)|classification|histopathology|10.1109/TMI.2024.3419697|
|[Retinal-Datasets](https://github.com/lxirich/MM-Retinal)|[google drive](https://drive.google.com/drive/folders/177RCtDeA6n99gWqgBS_Sw3WT6qYbzVmy)|classification|retinal|10.1007/978-3-031-72378-0_67|

### VLMs
### Appendix: non-medical VLMs corresponding medical VLMs
|model name|year|dataset|image encoder|text encoder|paper|model URL|
|:-:|:-:|:-:|:-:|:-:|:-|:-|
|CLIP|Feb. 21|LAION?|ViT|BERT|[PLMR](https://proceedings.mlr.press/v139/radford21a)|[github](https://github.com/openai/CLIP)|
|CoCa|May. 22|JFT-3B and ALIGN|||[arXiv](https://arxiv.org/abs/2205.01917)|[github](https://github.com/lucidrains/CoCa-pytorch)|
|DINOv2|Apr. 23|LVD-142M|ViT||[arXiv](https://arxiv.org/abs/2304.07193)|[github](https://github.com/facebookresearch/dinov2?tab=readme-ov-file)|
|LLaVA|Apr. 23|CC-595K subset and LLaVA-Instruct-158K dataset|||[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html), [CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.html)|[github](https://github.com/haotian-liu/LLaVA)|
|LLM2CLIP|Nov. 24|ShareCaptioner modified CC-3M dataset, Wikitext-103 dataset and the E5 dataset|EVA ViT L/14-224|Mistral Nemo 12B with LoRA finetuning|[arXiv](https://arxiv.org/abs/2411.04997)|[github](https://github.com/microsoft/LLM2CLIP)|
|ALIGN|Feb. 21|[alt-text](https://www.mdpi.com/2076-3417/13/19/11103?utm_source=chatgpt.com)|EfficientNet|BERT|[arXiv](https://arxiv.org/abs/2102.05918)|[github](https://github.com/ALIGN-analoglayout/ALIGN-public)|

### CLIP based models
|model name|year|dataset|image encoder|text encoder|Modality|paper|model URL|
|:-:|:-:|:-:|:-:|:-:|:-|:-|:-|
|BiomedCLIP|Mar. 23|PMC-15M|CLIP-ViT-B-16|PubMedBERT|all|[arXiv](https://arxiv.org/abs/2303.00915)|[huggng face](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)|
|UniMedCLIP|Dec. 24|[UniMed](https://github.com/mbzuai-oryx/UniMed-CLIP/blob/main/docs/UniMed-DATA.md)|CLIP-ViT-L-16 and L-14|BioMed-BERT|all|[arXiv](https://arxiv.org/abs/2412.10372)|[github](https://github.com/mbzuai-oryx/UniMed-CLIP)|
|CXR-CLIP|Oct. 23|MIMIC-CXR,  CheXpert,  ChestX-ray14, RSNA pneumonia, SIIM Pneumothorax, VinDR-CXR,  Open-I|ResNet50 or SwinTransformer|BioClinicalBERT|X-ray|[MICCAI2023](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_10)([arXiv](https://arxiv.org/abs/2310.13292))|[github](https://github.com/Soombit-ai/cxr-clip)|
|CONCH|Mar. 24|original (from PubMed)|CoCa based|CoCa based|histpathology|[nature medicine](https://www.nature.com/articles/s41591-024-02856-4)|[github](https://github.com/mahmoodlab/CONCH)|
|UNI|Mar. 24|Mass-100K<br>(100K WSI)<br>(over 75M images)|DINOv2|?|histpathology|[nature medicine](https://www.nature.com/articles/s41591-024-02857-3)|[github](https://github.com/mahmoodlab/UNI)|
|CHIEF|Sep. 24|original from [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), [GTEx](https://www.gtexportal.org/home/), PAIP, [PANDA](https://panda.grand-challenge.org/), BCC, BCNB, ACROBAT and TOC|original (CTransPath based)|CLIP|histpathology|[nature](https://www.nature.com/articles/s41586-024-07894-z)|[github](https://github.com/hms-dbmi/CHIEF)|
|CT-CLIP|Oct. 24|[CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)|3D transformer based on CT-ViT|CXR-Bert|CT (3D)|[arXiv](https://arxiv.org/abs/2403.17834)|[Hugging face](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)|
|CLIP-DR|Jul. 24|[GDR-Bench](https://github.com/chehx/DGDR/blob/main/GDRBench/README.md)|ResNet50|pre-trained Transformer|Retinopathy|[MICCAI2024](https://link.springer.com/chapter/10.1007/978-3-031-72378-0_62)([arXiv](https://arxiv.org/abs/2407.04068))|[github](https://github.com/Qinkaiyu/CLIP-DR)|
|VisionUnite|Aug. 24|[MMFundus](https://github.com/HUANGLIZI/MMFundus)|original (EVA02 and CLIP based)|llama-7B|Ophthalmology|[arXiv](https://arxiv.org/abs/2408.02865)|[github](https://github.com/HUANGLIZI/VisionUnite)|

### VQA based models
|model name|year|dataset|image encoder|text encoder|Modality|paper|model URL|
|:-:|:-:|:-:|:-:|:-:|:-|:-|:-|
|LLaVA-Med|Jun. 23|PMC-15M|CLIP-ViT-L/14-336|mistral-7b|all|[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html)|[github](https://github.com/microsoft/LLaVA-Med)|
|MedTrinity|Aug. 24|[MedTrinity-25M](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M)|same as LLaVA-Med|same as LLaVA-Med|all|[arXiv](https://www.arxiv.org/abs/2408.02900)|[github](https://github.com/UCSC-VLAA/MedTrinity-25M)|
|PathChat|Jun.24|original from PubMed Open Access, |UNI|Llama2LLM|Pathology|[nature](https://www.nature.com/articles/s41586-024-07618-3)|[github](https://github.com/fedshyvana/pathology_mllm_training)

## survey source
|name|URL|
|:-|:-|
|Data-Centric Foundation Models in Computational Healthcare|https://github.com/openmedlab/Data-Centric-FM-Healthcare|
|A Survey of Publicly Available MRI Datasets for Potential Use in Artificial Intelligence Research|https://pubmed.ncbi.nlm.nih.gov/37888298/|
|A survey on lung CT datasets and research trends|https://link.springer.com/article/10.1007/s42600-021-00138-3|
|Awesome-Healthcare-Foundation-Models|https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models|
|Medical Vision-and-Language Tasks and Methodologies: A Survey|https://github.com/YtongXie/Medical-Vision-and-Language-Tasks-and-Methodologies-A-Survey|
|Alzheimer's Disease Neuroimaging Initiative (ADNI)|https://pmc.ncbi.nlm.nih.gov/articles/PMC2809036/|
|Awesome Vision-Language Models (VLMs) for Medical Report Generation (RG) and Visual Question Answering (VQA)|https://github.com/lab-rasool/Awesome-Medical-VLMs-and-Datasets/tree/main|
