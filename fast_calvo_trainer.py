import cv2
import numpy as np
import os

import training_engine_sae as training

class FastCalvoTrainer:
    def __init__(self, inputs: dict, settings: dict, outputs: dict):
        self.settings = settings
        self.inputs = inputs
        self.outputs = outputs

    def run(self) -> bool:
        # Settings
        patch_height = self.settings['Patch height']
        patch_width = self.settings['Patch width']
        max_number_of_epochs = self.settings['Maximum number of training epochs']
        max_samples = self.settings['Maximum number of samples per label']

        # Inputs/Outputs
        
        # Shouldn't use codes, but the actual flag from the cv lib:
        #   https://docs.opencv.org/master/d8/d6a/group__imgcodecs__flags.html
        # Because not using flags breaks in newer versions of python CV2.
        # input_image = cv2.imread(self.inputs['Image'], True)
        input_image = cv2.imread(self.inputs['Image'], cv2.IMREAD_COLOR)

        # To keep the Alpha channel, we use cv2.IMREAD_UNCHANGED. 
        background = cv2.imread(self.inputs['Background'], cv2.IMREAD_UNCHANGED)
        regions = cv2.imread(self.inputs['Selected Regions'], cv2.IMREAD_UNCHANGED)
        output_models_path = {
            'background': self.outputs['Background Model'],
        }

        # create categorical ground-truth
        gt = {}
        regions_mask = (regions[:, :, 3] == 255)
        gt['background'] = (background[:, :, 3] == 255) # background is already restricted to the selected regions (based on Pixel.js' behaviour)

        # Optional layers
        input_ports = len([x for x in self.inputs if x[:17] == "rgba PNG - Layer "])
        for port_number in range(input_ports):
            layer = 'rgba PNG - Layer %d' % port_number

            # Check if file exist (throws error if it doesn't)
            with open(self.inputs[layer]) as f:
                pass

            file_ = cv2.imread(self.inputs[layer], cv2.IMREAD_UNCHANGED)
            mask = (file_[:, :, 3] == 255)
            gt[layer] = np.logical_and(mask, regions_mask)
            output_models_path[layer] = self.outputs["Model %d" % port_number]

        # Call in training function
        status = training.train_msae(
            input_image=input_image,
            gt=gt,
            height=patch_height,
            width=patch_width,
            output_path=output_models_path,
            epochs=max_number_of_epochs,
            max_samples_per_class=max_samples
        )

        print('Finishing the Fast CM trainer job.')
        return True
