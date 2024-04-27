from torch import nn

class FewShotModel(nn.Module):
    """
    Few-Shot Learning Model
    """
    def __init__(self, baseModel, num_classes):
        super(FewShotModel, self).__init__()
        
        self.num_classes = num_classes
        # load the base model
        self.model = baseModel

        self.regressor = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
        	nn.Linear(self.model.fc.in_features, 512),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(512, 512),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(512, self.num_classes)
        )       

        # set the classifier of our base model to produce outputs features
        # from the last convolution block
        self.model.fc.in_features = nn.Identity()
    
    def forward(self, x):

        features = self.model(x)
        bboxes = self.regressor(features)
        classes = self.classifier(features)
        return (bboxes, classes)
