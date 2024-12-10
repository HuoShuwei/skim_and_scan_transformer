
# Skim-and-Scan Transformer (SSTr)

Official PyTorch implementation of "Skim-and-Scan Transformer: A New Transformer-inspired Architecture for Video-Query based Video Moment Retrieval"

## Note

🚧  **Current Status** :

* The network architecture implementation has been released
* The pre-trained models will be available soon

## Overview

This repository contains the implementation of Skim-and-Scan Transformer (SSTr), a novel transformer-inspired architecture for Video-Query based Video Moment Retrieval (VQ-VMR). SSTr is designed to effectively locate a target segment in a long untrimmed video that semantically corresponds to a given query video clip.

### Key Features

* **Dual Spatiotemporal Encoders** : Constructs rich visual and temporal semantic representations for video frames
* **Multi-level Skim-and-Scan Modules** : Enables feature interactions across multiple temporal granularities
* **Dual Prediction Heads** : Combines end-to-end and frame-wise predictions for accurate boundary localization
