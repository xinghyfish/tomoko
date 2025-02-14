import sys
from lang_sam import LangSAM


def main():
    model_type = sys.argv[1]
    full_model_type = "sam2.1_hiera_%s" % model_type
    model_path = "configs/sam2.1/sam2.1_hiera_%s.pt" % model_type
    model = LangSAM(full_model_type, model_path)
