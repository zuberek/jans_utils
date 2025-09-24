import random
import os
import sys
from pathlib import Path
from typing import List, Iterable, Optional, Dict, Any, Tuple, Union
import copy

import tensorflow as tf

from octopus.eval import EvalConfig
from octopus.train import TrainConfig
from octopus.params import Params
from octopus.params_combinations import ParamsCombinations
from octopus.training_params import TrainingExperimentParams
from octopus.common_exp_params import CommonExpParams
from octopus.experiment_db_api.database_api import ExperimentDatabase
from octopus.evaluation_params import EvaluationExperimentParams
from octopus.train_early_stopping import EarlyStopping


from octopus.pipelines.preprocess import PreprocessPipeline as PreprocessPipelineOctopus
from octopus.pipelines.batching import CroppingBatcher2D
from octopus.pipelines.neighboring_channel_concatenator import NeighboringChannelConcatenator, PaddingMethod

from octopus.losses.tversky import Tversky
from octopus.utils.tensor_type_hints import AF_nhwc, AF_nc

from framework.models.unet import create_unet_generic
from framework.data.pipelines.cache import Cache
from framework.models.normalization import BatchNormFactory
from framework.data.pipelines.load import Load
from framework.data.pipelines.save import Save
from framework.data.pipelines.copy_pipeline import Copy
from framework.data.pipelines.batch import BatchInput 
from framework.data.preprocessing.pad_to_factor import PadToFactor
from framework.data.pipelines.batch_exams import BatchExams
from framework.data.pipelines.dump import Dump
from framework.data.pipelines.map import Map
from framework.data.pipelines.preprocess import PreprocessPipeline
from framework.data.pipelines.visualize import VisualizeHWCTensor, VisualizeBorders
from framework.data.pipelines.drop import Drop
from framework.data.pipelines.unbatch import Unbatch
from framework.data.pipelines.shuffle import Shuffle
from framework.data.pipelines.print import Print
from framework.data.preprocessing.resize import Resize
from framework.data.pipelines.filter import Filter
from framework.data.pipelines.keep import Keep
from framework.losses.dice import Dice

from denoise.denoiser import DenoiserModule

from framework.tests_system.test_retina_roi_detector import MODEL_PATH as RETINA_ROI_DETECTOR_MODEL_PATH
import framework.data.pipe_params_per_graph.posterior_roi_detector_params as posterior_roi_params
import framework.data.pipe_params_per_graph.posterior_params as posterior_params
from framework.data.preprocessing.strip_straightener import StripStraightener
from framework.data.pipelines.strip_height_calculator import StripHeightCalculator
from framework.data.pipelines.tissue_detector import TissueDetector

from framework.data.pipelines.augment import AugmentSequential
from framework.augment.augment_prob import AugmentProb
from framework.augment.augmentations import HorizontalMirroring
from framework.augment.augmentations import XZScaling
from framework.augment.augmentations import AdjustBrightness
from framework.augment.augmentations import AdjustContrast
from framework.augment.augmentations import GaussianNoise
from framework.augment.augmentations import GammaCorrection
from framework.augment.augmentations import ShearAndCrop
from framework.augment.augmentations import ShiftAscans
from framework.augment.warping import Warping
from framework.augment.elastic_deformation import ElasticDeformation
