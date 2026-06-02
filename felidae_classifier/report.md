#Tests :

- Flatten images : MLP(3074, 16, 3) with 600 samples, learning rate of 0.01 and a train/test split of 80% train / 20% test -> 75% accuracy train, 35% accuracy test
- Flatten images : MLP(3074, 16, 3) with 29000 samples, learning rate of 0.01 and a train/test split of 80% train / 20% test -> 46% accuracy train, 46% accuracy test
- Flatten images and extracting features : MLP(20, 16, 3) with 9000 samples, learning rate of 0.01 and a train/test split of 80% train / 20% test -> 44% to 48% accuracy train, 44% to 48% accuracy train
- Flatten images and extracting features : MLP(20, 16, 16, 3) with 9000 samples, learning rate of 0.01 and a train/test split of 80% train / 20% test -> 44% to 48% accuracy train, 44% to 48% accuracy train

