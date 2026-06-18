import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules", "ssl-singer-identity"))

from singer_identity import load_model as load_singer_identity_model

__all__ = ["load_singer_identity_model"]
