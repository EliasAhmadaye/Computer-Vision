# Image Dataset Management – Computer Vision

Ce projet consiste à développer un module simple pour **gérer des jeux de données d’images** utilisés en vision par ordinateur (classification, détection, segmentation).

Le travail est divisé en deux parties.

La **première partie** repose sur des **fonctions Python indépendantes** (sans classes) permettant :
- de lister et lire des images,
- de les afficher,
- de les redimensionner et normaliser,
- d’appliquer des augmentations (flip, crop, luminosité),
- de mélanger, diviser, sauvegarder et charger un dataset.

La **seconde partie** introduit une classe `ImageDataset` qui regroupe ces fonctionnalités afin de manipuler un jeu de données comme un objet unique.  
Cette classe permet d’accéder aux images par index, d’afficher une ou plusieurs images, de mélanger et découper le dataset, de le sauvegarder, et de le comparer à d’autres datasets.

Ce projet permet de comprendre les bases de la gestion des datasets en vision par ordinateur avant de passer à des implémentations plus avancées.

Projet réalisé dans le cadre d’un cours de **Computer Vision**.
