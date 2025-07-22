# <p align=center>`Awesome Mamba`</p> # 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

:fire::fire: This is a collection of awesome articles about Mamba models (With a particular emphasis on Medical Image Analysis):fire::fire:

- Our survey paper on arXiv: [Computation-Efficient Era: A Comprehensive Survey of State Space Models in Medical Image Analysis](https://arxiv.org/abs/2406.03430) :heart:

#### Citation
```python
@article{heidari2024computation,
  title={Computation-Efficient Era: A Comprehensive Survey of State Space Models in Medical Image Analysis},
  author={Heidari, Moein and Ghorbani Kolahi, Sina and Karimijafarbigloo, Sanaz and Azad, Bobby and Bozorgpour, Afshin and Hatami, Soheila and Azad, Reza and Diba, Ali and Bagci, Ulas and Merhof, Dorit and others},
  journal={arXiv e-prints},
  pages={arXiv--2406},
  year={2024}
}
```

## Updates
- :sunglasses: **First release:** June 05, 2024


## Table of Contents
- [Survey Papers](#surveys)
- [Architecture Redesign](#architecture-redesign)
- [Remote Sensing](#remote-sensing)
- [Speech Processing](#speech-processing)
- [Video Processing](#video-processing)
- [Activity Recognition](#activity-recognition)
- [Image Enhancement](#image-enhancement)
- [Image & Video Generation](#image-video-generation)
- [Medical Imaging](#medical-imaging)
- [Image Segmentation](#image-segmentation)
- [Reinforcement Learning](#reinforcement-learning)
- [Natural Language Processing](#natural-language-processing)
- [3D Recognition](#3d-recognition)
- [Multi-Modal Understanding](#multi-modal-understanding)
- [Time Series](#time-series)
- [GNN](#gnn)
- [Point Cloud](#point-cloud)
- [Tabular Data](#tabular-data)
- [Other Applications](#other-applications)


## Surveys

- [From Generalization Analysis to Optimization Designs for State Space Models](https://arxiv.org/abs/2405.02670) 
- [MambaOut: Do We Really Need Mamba for Vision?](https://arxiv.org/abs/2405.07992) [[Github](https://github.com/yuweihao/MambaOut)]
- [State Space Model for New-Generation Network Alternative to Transformers: A Survey](https://arxiv.org/abs/2404.09516) [[Github](https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List)]
- [A Survey on Visual Mamba](https://arxiv.org/abs/2404.15956)
- [Mamba-360: Survey of State Space Models as Transformer Alternative for Long Sequence Modelling: Methods, Applications, and Challenges](https://arxiv.org/abs/2404.16112) [[Github](https://github.com/badripatro/mamba360)]
- [Vision Mamba: A Comprehensive Survey and Taxonomy](https://arxiv.org/abs/2405.04404) [[Github](https://github.com/lx6c78/Vision-Mamba-A-Comprehensive-Survey-and-Taxonomy)]
- [A Survey on Vision Mamba: Models, Applications and Challenges](https://arxiv.org/abs/2404.18861) [[Github](https://github.com/Ruixxxx/Awesome-Vision-Mamba-Models)]
  
## Preliminaries and State Space Models
- [S7: Selective and Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2410.03464)
- [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) [[Github](https://github.com/HazyResearch/hippo-code)]
- [S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396v3) [[Github](https://github.com/state-spaces/s4)]
- [H3: Hungry Hungry Hippos: Toward Language Modeling with State Space Models](https://arxiv.org/abs/2212.14052) [[Github](https://github.com/HazyResearch/H3)]
- [LOCOST: State-Space Models for Long Document Abstractive Summarization](https://arxiv.org/abs/2401.17919) [[Github](https://github.com/flbbb/locost-summarization)]
- [Theoretical Foundations of Deep Selective State-Space Models](https://arxiv.org/abs/2402.19047)
- [S4++: Elevating Long Sequence Modeling with State Memory Reply](https://openreview.net/forum?id=bdnw4qjfH9)
- [Hieros: Hierarchical Imagination on Structured State Space Sequence World Models](https://openreview.net/forum?id=5j6wtOO6Fk) [[Github](https://github.com/snagnar/hieros)]
- [State Space Models as Foundation Models: A Control Theoretic Overview](https://arxiv.org/abs/2403.16899)
- [Selective Structured State-Spaces for Long-Form Video Understanding](https://arxiv.org/abs/2303.14526)
- [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)[[Github](https://github.com/microsoft/unilm/tree/master/retnet)]
- [Convolutional State Space Models for Long-Range Spatiotemporal Modeling](https://arxiv.org/abs/2310.19694)[[Github](https://github.com/NVlabs/ConvSSM)]
- [Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions](https://arxiv.org/abs/2310.18780)[[Github](https://github.com/togethercomputer/stripedhyena)]
- [Resurrecting Recurrent Neural Networks for Long Sequences](https://icml.cc/virtual/2023/oral/25438)
- [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866v3.pdf)[[Github](https://github.com/hazyresearch/safari)]
  
## Architecture Redesign

- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) [[Github](https://github.com/state-spaces/mamba)]
- [Mamba: Linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752) [[Github](https://github.com/state-spaces/mamba)]
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) [[Github](https://github.com/state-spaces/mamba)]
- [Locating and Editing Factual Associations in Mamba](https://www.arxiv.org/abs/2404.03646) [[Github](https://github.com/arnab-api/romba)]
- [MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection](https://arxiv.org/abs/2403.19888) [[Github](https://github.com/MambaMixer/M2)]
- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887v1)
- [Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data](https://arxiv.org/abs/2402.05892)[[Github](https://github.com/jacklishufan/Mamba-ND)]
- [Incorporating Exponential Smoothing into MLP: A Simple but Effective Sequence Model](https://arxiv.org/abs/2403.17445v1) [[Github](https://github.com/pkuai-lingroup/etsmlp)]
- [PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition](https://arxiv.org/abs/2403.17695v1) [[Github](https://github.com/chenhongyiyang/plainmamba)]
- [Understanding Robustness of Visual State Space Models for Image Classification](https://arxiv.org/abs/2403.10935v1)
- [Efficientvmamba: Atrous selective scan for light weight visual mamba](https://arxiv.org/abs/2403.09977v1) [[Github](https://github.com/terrypei/efficientvmamba)]
- [Localmamba: Visual state space model with windowed selective scan](https://arxiv.org/abs/2403.09338v1) [[Github](https://github.com/hunto/localmamba)]
- [Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models](https://arxiv.org/abs/2403.03900v1) [[Github](https://github.com/chengkai-liu/mamba4rec)]
- [The hidden attention of mamba models](https://arxiv.org/abs/2403.01590v2) [[Github](https://github.com/ameenali/hiddenmambaattn)] 
- [Learning method for S4 with Diagonal State Space Layers using Balanced Truncation](https://arxiv.org/abs/2402.15993v2)
- [BlackMamba: Mixture of Experts for State-Space Models](https://arxiv.org/abs/2402.01771)[[Github](https://github.com/Zyphra/BlackMamba)]
- [MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts](https://arxiv.org/abs/2401.04081)[[Github](https://github.com/kyegomez/MoE-Mamba)]
- [Scalable Diffusion Models with State Space Backbone](https://arxiv.org/abs/2402.05608v3) [[Github](https://github.com/feizc/dis)]
- [ZigMa: Zigzag Mamba Diffusion Model](https://arxiv.org/abs/2403.13802)[[Github](https://taohu.me/zigma/)]
- [Spectral State Space Models](https://arxiv.org/abs/2312.06837v3)
- [Mamba-unet: Unet-like pure visual mamba for medical image segmentation](https://arxiv.org/abs/2402.05079v2) [[Github](https://github.com/ziyangwang007/mamba-unet)]
- [Mambabyte: Token-free selective state space model](https://arxiv.org/abs/2401.13660v2)
- [Vmamba: Visual state space model](https://arxiv.org/abs/2401.10166v1) [[Github](https://github.com/mzeromiko/vmamba)]
- [Vision mamba: Efficient visual representation learning with bidirectional state space model](https://arxiv.org/abs/2401.09417v2) [[Github](https://github.com/hustvl/vim)]
- [Sparsified State-Space Models are Efficient Highway Networks](https://arxiv.org/abs/2505.20698) [[Github](https://github.com/woominsong/Simba)]

## Remote Sensing
- [ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model](https://arxiv.org/abs/2404.03425v1) [[Github](https://github.com/chenhongruixuan/mambacd)]
- [RS-Mamba for Large Remote Sensing Image Dense Prediction](https://arxiv.org/abs/2404.02668v1) [[Github](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)]
- [RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation](https://arxiv.org/abs/2404.02457v1) [[Github](https://github.com/sstary/ssrs)]
- [HSIMamba: Hyperspectral Imaging Efficient Feature Learning with Bidirectional State Space for Classification](https://arxiv.org/abs/2404.00272) [[Github](https://github.com/Judyxyang/judyxyang/blob/master/HSi_UH2013_P7_AB_VIM_V3_6_0330.ipynb)]
- [Rsmamba: Remote sensing image classification with state space model](https://arxiv.org/abs/2403.19654) [[Github](https://github.com/KyanChen/RSMamba)]
- [HSIMamba: Hyperpsectral Imaging Efficient Feature Learning with Bidirectional State Space for Classification](https://arxiv.org/abs/2404.00272) [[Github](https://github.com/Judyxyang/judyxyang/blob/master/HSi_UH2013_P7_AB_VIM_V3_6_0330.ipynb)]
- [Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model](https://arxiv.org/abs/2404.01705) [[Github](https://github.com/zhuqinfeng1999/Samba)]

## Speech Processing
- [SPMamba: State-space model is all you need in speech separation](https://arxiv.org/abs/2404.02063) [[Github](https://github.com/JusperLee/SPMamba)]
- [Dual-path Mamba: Short and Long-term Bidirectional Selective Structured State Space Models for Speech Separation](https://arxiv.org/abs/2403.18257v1) [[Github](https://github.com/xi-j/Mamba-TasNet)]

## Video Processing
- [SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding](https://arxiv.org/abs/2404.01174v1)
- [Video mamba suite: State space model as a versatile alternative for video understanding](https://arxiv.org/abs/2403.09626v1) [[Github](https://github.com/opengvlab/video-mamba-suite)]
- [SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces](https://arxiv.org/abs/2403.07711v3) [[Github](https://github.com/shim0114/ssm-meets-video-diffusion-models)]
- [Videomamba: State space model for efficient video understanding](https://arxiv.org/abs/2403.06977) [[Github](https://github.com/OpenGVLab/VideoMamba)]

## Activity Recognition
- [HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM](https://arxiv.org/abs/2403.20183v1)
- [VMRNN: Integrating Vision Mamba and LSTM for Efficient and Accurate Spatiotemporal Forecasting](https://arxiv.org/abs/2403.16536)[[Github](https://github.com/yyyujintang/VMRNN-PyTorch)]

## Image Enhancement
- [Aggregating Local and Global Features via Selective State Spaces Model for Efficient Image Deblurring](https://arxiv.org/abs/2403.20106)
- [Serpent: Scalable and Efficient Image Restoration via Multi-scale Structured State Space Models](https://arxiv.org/abs/2403.17902v1)
- [VmambaIR: Visual State Space Model for Image Restoration](https://arxiv.org/abs/2403.11423v1) [[Github](https://github.com/alphacatplus/vmambair)]
- [Activating Wider Areas in Image Super-Resolution](https://arxiv.org/abs/2210.03743v1) [[Github](https://github.com/george-gca/sr-pytorch-lightning)]
- [MambaIR: A Simple Baseline for Image Restoration with State-Space Model](https://arxiv.org/abs/2402.15648v2) [[Github](https://github.com/csguoh/mambair)]
- [Pan-Mamba: Effective pan-sharpening with State Space Model](https://arxiv.org/abs/2402.12192v2) [[Github](https://github.com/alexhe101/pan-mamba)]
- [U-shaped Vision Mamba for Single Image Dehazing](https://arxiv.org/abs/2402.04139v4) [[Github](https://github.com/zzr-idam/uvm-net)]

## Image-Video-Generation
- [DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis](https://arxiv.org/abs/2405.14224) [[Github](https://github.com/tyshiwo1/DiM-DiffusionMamba/)]

## Medical Imaging
- [MambaRoll: Physics-Driven Autoregressive State Space Models for Medical Image Reconstruction](https://arxiv.org/abs/2412.09331) [[GitHub](https://github.com/icon-lab/MambaRoll)]
- [DenoMamba: A fused state-space model for low-dose CT denoising](https://www.arxiv.org/abs/2409.13094) [[Github](https://github.com/icon-lab/DenoMamba)]
- [Vim4Path: Self-Supervised Vision Mamba for Histopathology Images](https://arxiv.org/abs/2404.13222) [[Github](https://github.com/AtlasAnalyticsLab/Vim4Path)]
- [VMambaMorph: a Visual Mamba-based Framework with Cross-Scan Module for Deformable 3D Image Registration](https://arxiv.org/abs/2404.05105) [[Github](https://github.com/ziyangwang007/VMambaMorph)]
- [UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation](https://arxiv.org/abs/2403.20035v1) [[Github](https://github.com/wurenkai/UltraLight-VM-UNet)]
- [Rotate to Scan: UNet-like Mamba with Triplet SSM Module for Medical Image Segmentation](https://arxiv.org/abs/2403.17701v2) 
- [Integrating Mamba Sequence Model and Hierarchical Upsampling Network for Accurate Semantic Segmentation of Multiple Sclerosis Legion](https://arxiv.org/abs/2403.17432v1) 
- [CMViM: Contrastive Masked Vim Autoencoder for 3D Multi-modal Representation Learning for AD classification](https://arxiv.org/abs/2403.16520v1) 
- [H-vmunet: High-order vision mamba unet for medical image segmentation](https://arxiv.org/abs/2403.13642v1) [[Github](https://github.com/wurenkai/h-vmunet)]
- [ProMamba: Prompt-Mamba for polyp segmentation](https://arxiv.org/abs/2403.13660v2)
- [Vm-unet-v2 rethinking vision mamba unet for medical image segmentation](https://arxiv.org/abs/2403.09157v1) [[Github](https://github.com/nobodyplayer1/vm-unetv2)]
- [MD-Dose: A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction](https://arxiv.org/abs/2403.08479v1) [[Github](https://github.com/flj19951219/mamba_dose)]
- [Large Window-based Mamba UNet for Medical Image Segmentation: Beyond Convolution and Self-attention](https://arxiv.org/abs/2403.07332v1) [[Github](https://github.com/wjh892521292/lma-unet)]
- [MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology](https://arxiv.org/abs/2403.06800v1) [[Github](https://github.com/isyangshu/mambamil)]
- [Clinicalmamba: A generative clinical language model on longitudinal clinical notes](https://arxiv.org/abs/2403.05795v1) [[Github](https://github.com/whaleloops/clinicalmamba)]
- [Lightm-unet: Mamba assists in lightweight unet for medical image segmentation](https://arxiv.org/abs/2403.05246v2) [[Github](https://github.com/mrblankness/lightm-unet)]
- [MedMamba: Vision Mamba for Medical Image Classification](https://arxiv.org/abs/2403.03849v3) [[Github](https://github.com/YubiaoYue/MedMamba)]
- [Weak-Mamba-UNet: Visual Mamba Makes CNN and ViT Work Better for Scribble-based Medical Image Segmentation](https://arxiv.org/abs/2402.10887v1) [[Github](https://github.com/ziyangwang007/mamba-unet)]
- [P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation](https://arxiv.org/abs/2402.08506v2)
- [Semi-Mamba-UNet: Pixel-Level Contrastive Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2402.07245v2) [[Github](https://github.com/ziyangwang007/mamba-unet)]
- [FD-Vision Mamba for Endoscopic Exposure Correction](https://arxiv.org/abs/2402.06378v2) [[Github](https://github.com/zzr-idam/FDVM-Net)]
- [Swin-umamba: Mamba-based unet with imagenet-based pretraining](https://arxiv.org/abs/2402.03302v2) [[Github](https://github.com/jiarunliu/swin-umamba)]
- [Vm-unet: Vision mamba unet for medical image segmentation](https://arxiv.org/abs/2402.02491v1)[[Github](https://github.com/jcruan519/vm-unet)]
- [Vivim: a video vision mamba for medical video object segmentation](https://arxiv.org/abs/2401.14168v3) [[Github](https://github.com/scott-yjyang/vivim)]
- [Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation](https://arxiv.org/abs/2401.13560v3) [[Github](https://github.com/ge-xing/segmamba)]
- [T-Mamba: Frequency-Enhanced Gated Long-Range Dependency for Tooth 3D CBCT Segmentation](https://arxiv.org/abs/2404.01065) [[Github](https://github.com/isbrycee/T-Mamba)]
- [U-mamba: Enhancing long-range dependency for biomedical image segmentation](https://arxiv.org/abs/2401.04722v1) [[Github](https://github.com/bowang-lab/U-Mamba)]
- [MambaMorph: a Mamba-based Backbone with Contrastive Feature Learning for Deformable MR-CT Registration](https://arxiv.org/pdf/2401.13934.pdf)[[Github](https://github.com/Guo-Stone/MambaMorph)]
- [nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model](https://arxiv.org/abs/2402.03526)[[Github](https://github.com/lhaof/nnMamba)]
- [MambaMIR: An Arbitrary-Masked Mamba for Joint Medical Image Reconstruction and Uncertainty Estimation](https://arxiv.org/abs/2402.18451)[[Github](https://github.com/ayanglab/MambaMIR)]
- [ViM-UNet: Vision Mamba for Biomedical Segmentation](https://arxiv.org/abs/2404.07705)[[Github](https://github.com/constantinpape/torch-em/blob/main/vimunet.md)]
- [VM-DDPM: Vision Mamba Diffusion for Medical Image Synthesis](https://arxiv.org/abs/2405.05667)
- [HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation](https://arxiv.org/abs/2405.05007)
- [I2I-Mamba: Multi-modal Medical Image Synthesis via Selective State Space Modeling](https://arxiv.org/abs/2405.14022) [[GitHub](https://github.com/icon-lab/I2I-Mamba)]
- [MUCM-Net: A Mamba Powered UCM-Net for Skin Lesion Segmentation](https://arxiv.org/abs/2405.15925) [[Github](https://github.com/chunyuyuan/MUCM-Net)]
- [Cardiovascular Disease Detection from Multi-View Chest X-rays with BI-Mamba](https://arxiv.org/abs/2405.18533v1)
- [Convolution and Attention-Free Mamba-based Cardiac Image Segmentation](https://arxiv.org/abs/2406.05786)
- [AC-MAMBASEG: An adaptive convolution and Mamba-based architecture for enhanced skin lesion segmentation](https://arxiv.org/abs/2405.03011) [[Github](https://github.com/vietthanh2710/AC-MambaSeg)]
- [SAM-VMNet: Deep Neural Networks For Coronary Angiography Vessel Segmentation](https://arxiv.org/abs/2406.00492)
- [MHS-VM: Multi-Head Scanning in Parallel Subspaces for Vision Mamba](https://arxiv.org/abs/2406.05992v1)[[Github](https://github.com/PixDeep/MHS-VM/tree/main)]
  
## Image Segmentation
- [UU-Mamba: Uncertainty-aware U-Mamba for Cardiac Image Segmentation](https://arxiv.org/abs/2405.17496) [[Github](https://github.com/tiffany9056/UU-Mamba)]
- [Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation](https://arxiv.org/abs/2404.04256) [[Github](https://github.com/zifuwan/Sigma)]

## Reinforcement Learning
- [Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2403.19925v1) [[Github](https://github.com/toshihiro-ota/decision-mamba)]
- [MAMBA: an Effective World Model Approach for Meta-Reinforcement Learning](https://arxiv.org/abs/2403.09859) [[Github](https://github.com/zoharri/mamba)]

## Natural Language Processing
- [RankMamba Benchmarking Mamba's Document Ranking Performance in the Era of Transformers](https://arxiv.org/abs/2403.18276v1) [[Github](https://github.com/zhichaoxu-shufe/rankmamba)]
- [Densemamba: State space models with dense hidden connection for efficient large language models](https://arxiv.org/abs/2403.00818v2) [[Github](https://github.com/wailordhe/densessm)]
- [Is Mamba Capable of In-Context Learning?](https://arxiv.org/abs/2402.03170v1)
- [BioMamba: A Pre-trained Biomedical Language Representation Model Leveraging Mamba](https://arxiv.org/abs/2408.02600)

## 3D Recognition
- [Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction](https://arxiv.org/abs/2403.18795v2)
- [Motion mamba: Efficient and long sequence motion generation with hierarchical and bidirectional selective ssm](https://arxiv.org/abs/2403.07487v3) [[Github](https://github.com/steve-zeyu-zhang/MotionMamba/)]

## Multi-Modal Understanding
- [Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models](https://arxiv.org/abs/2405.15574)[[Github](https://github.com/ByungKwanLee/Meteor)]
- [ReMamber: Referring Image Segmentation with Mamba Twister](https://arxiv.org/abs/2403.17839v1)
- [VL-Mamba: Exploring State Space Models for Multimodal Learning](https://arxiv.org/abs/2403.13600v1)
- [Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference](https://arxiv.org/abs/2403.14520)[[Github](https://github.com/h-zhao1997/cobra)]
- [SurvMamba: State Space Model with Multi-grained Multi-modal
Interaction for Survival Prediction](https://arxiv.org/abs/2404.08027)
- [MambaDFuse: A Mamba-based Dual-phase Model forMulti-modality Image Fusion](https://arxiv.org/abs/2404.08406)
- [APRICOT-Mamba: Acuity Prediction in Intensive Care Unit (ICU): Development and Validation of a Stability, Transitions, and Life-Sustaining Therapies Prediction Model](https://arxiv.org/abs/2311.02026)

## Time Series
- [SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series](https://arxiv.org/abs/2403.15360v1) [[Github](https://github.com/badripatro/simba)]
- [Is Mamba Effective for Time Series Forecasting?](https://arxiv.org/abs/2403.11144v2) [[Github](https://github.com/wzhwzhwzh0921/s-d-mamba)]
- [TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting](https://arxiv.org/abs/2403.09898v1) [[Github](https://github.com/atik-ahamed/timemachine)]
- [MambaStock: Selective state space model for stock prediction](https://arxiv.org/abs/2402.18959v1) [[Github](https://github.com/zshicode/mambastock)]
- [Bi-Mamba4TS: Bidirectional Mamba for Time Series Forecasting](https://arxiv.org/abs/2404.15772)

## GNN
- [STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model](https://arxiv.org/abs/2403.12418v2)
- [Graph Mamba: Towards Learning on Graphs with State Space Models](https://arxiv.org/abs/2402.08678v2) [[Github](https://github.com/graphmamba/gmn)]
- [Graph-Mamba: Towards long-range graph sequence modeling with selective state spaces](https://arxiv.org/abs/2402.00789v1) [[Github](https://github.com/bowang-lab/Graph-Mamba)]
- [Recurrent Distance Filtering for Graph Representation Learning](https://arxiv.org/abs/2312.01538)[[Github](https://github.com/skeletondyh/GRED)]
- [Modeling multivariate biosignals with graph neural networks and structured state space models](https://arxiv.org/abs/2211.11176)[[Github](https://github.com/tsy935/graphs4mer)]

## Point Cloud
- [Point mamba: A novel point cloud backbone based on state space model with octree-based ordering strategy](https://arxiv.org/abs/2403.06467v2) [[Github](https://github.com/irmvlab/point-mamba)]
- [Point Could Mamba: Point Cloud Learning via State Space Model](https://arxiv.org/abs/2403.00762v2) [[Github](https://github.com/zhang-tao-whu/pcm)]
- [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739v3) [[Github](https://github.com/lmd0311/pointmamba)]
- [3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering](https://arxiv.org/abs/2404.05522)
- [3DMambaComplete: Exploring Structured State Space Model for Point Cloud Completion](https://arxiv.org/abs/2404.07106)

## Tabular Data
- [MambaTab: A Simple Yet Effective Approach for Handling Tabular Data](https://arxiv.org/abs/2401.08867v1)
- [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291) [[Github](https://github.com/basf/mamba-tabular)]

## Other Applications
- [State Space Models for Event Cameras](https://arxiv.org/abs/2402.15584) [[Github](https://github.com/uzh-rpg/ssms_event_cameras)]
- [Mamba Drafters for Speculative Decoding](https://arxiv.org/abs/2506.01206)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xmindflow/Awesome_Mamba&type=Date)](https://star-history.com/#xmindflow/Awesome_Mamba&Date)
