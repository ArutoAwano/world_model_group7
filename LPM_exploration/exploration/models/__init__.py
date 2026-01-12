from .lbs import LBS
from .improve import LearningProgressCuriosity
from .ama import AMAPix2PixCuriosity
from .unet_improve import UNetLearningProgressCuriosity
from .RND import RandomNetworkDistillationCuriosity
from .icm import IntrinsicCuriosityModuleCuriosity
from .ensemble import EnsembleDisagreementCuriosity
from .tdd import TDDNetwork, TemporalDistanceDensityCuriosity
# from .tdd2 import TDDNetwork2, TemporalDistanceDensityCuriosity2

print("=== NEW CODE VERSION ===")  # Add this as first line in __init__