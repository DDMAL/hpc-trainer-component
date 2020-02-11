import cv2
import numpy as np
import os

import training_engine_sae as training

class FastCalvoTrainer:
    def __init__(self, inputs, settings, outputs):
        self.settings = settings
        self.inputs = inputs
        self.outputs = outputs

    def run(self):
        input_image = cv2.imread(self.inputs['Image'], True)
        background = cv2.imread(self.inputs['Background'], cv2.IMREAD_UNCHANGED)
        notes = cv2.imread(self.inputs['Music Layer'], cv2.IMREAD_UNCHANGED)
        lines = cv2.imread(self.inputs['Staff Layer'], cv2.IMREAD_UNCHANGED)
        text = cv2.imread(self.inputs['Text'], cv2.IMREAD_UNCHANGED)
        regions = cv2.imread(self.inputs['Selected Regions'], cv2.IMREAD_UNCHANGED)

        # create categorical ground-truth
        gt = {}
        regions_mask = (regions[:, :, 3] == 255)

        notes_mask = (lines[:, :, 3] == 255)
        gt['symbols'] = np.logical_and(notes_mask, regions_mask) #restrict layer to only the notes in the selected regions

        lines_mask = (lines[:, :, 3] == 255)
        gt['staff'] = np.logical_and(lines_mask, regions_mask) # restrict layer to only the staff lines in the selected regions

        text_mask = (text[:, :, 3] == 255)
        gt['text'] = np.logical_and(text_mask, regions_mask) # restrict layer to only the text in the selected regions

        gt['background'] = (background[:, :, 3] == 255) # background is already restricted to the selected regions (based on Pixel.js' behaviour)

        # Settings
        patch_height = self.settings['Patch height']
        patch_width = self.settings['Patch width']
        max_number_of_epochs = self.settings['Maximum number of training epochs']

        output_models_path = { 'background': self.outputs['Background Model'],
                        'symbols': self.outputs['Music Symbol Model'],
                        'staff': self.outputs['Staff Lines Model'],
                        'text': self.outputs['Text Model']
                        }

        # Call in training function
        status = training.train_msae(input_image,gt,
                                      height=patch_height,
                                      width=patch_width,
                                      output_path=output_models_path,
                                      epochs=max_number_of_epochs)

        print('Finishing the Fast CM trainer job.')
        return True
