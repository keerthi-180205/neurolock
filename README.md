# NeuroLock

A CNN-based face authentication system built from scratch for secure local device access.

## Overview

NeuroLock is a local face authentication system designed to authenticate a user based on their facial features.

The core deep-learning model is a custom CNN built from scratch using PyTorch. The project is being developed as a complete machine-learning pipeline, starting from data validation and preprocessing and progressing toward face representation, enrollment, verification, and authentication.

The system is designed with a clear separation between model training and the final face verification process.

## Problem Statement

Traditional authentication methods such as passwords and PINs require users to remember and enter credentials.

NeuroLock aims to provide a local, face-based authentication mechanism that can verify whether a captured face belongs to an enrolled identity.

The project focuses on building the core face recognition and authentication pipeline without relying on a pretrained face-embedding model for the core CNN.

## Project Goal

The goal of NeuroLock is to build a complete local face authentication system that can:

- Validate and organize face data
- Split identities into training, validation, and testing datasets
- Train a custom CNN for identity classification
- Learn meaningful facial representations
- Enroll authorized users
- Compare a new face against an enrolled representation
- Authenticate or reject the user based on a selected threshold

## Current Status

### Completed

- [] Project structure
- [x] Dataset validation
- [x] Identity-level dataset splitting
- [x] Custom PyTorch Dataset
- [x] DataLoader setup
- [x] Image preprocessing and transformations
- [x] CNN architecture implementation
- [x] CNN forward-pass testing

### In Progress

- [ ] CNN training
- [ ] Training/validation monitoring
- [ ] Best-model checkpointing
- [ ] Face embedding / representation extraction

### Upcoming

- [ ] User enrollment
- [ ] Face verification
- [ ] Authentication threshold selection
- [ ] Final test evaluation
- [ ] Local authentication integration

## Project Structure

```text
neurolock/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── logs/
│   ├── data_splitter.log
│   └── data_validator.log
│
├── src/
│   ├── data_splitter.py
│   ├── data_validator.py
│   ├── split_validator.py
│   ├── datasets.py
│   ├── model.py
│   ├── test_model.py
│   └── train.py
│
├── .gitignore
├── LICENSE
└── README.md