from PIL import Image
from lang_sam import LangSAM

from perception.vision.image_utils import show_masks_on_image

if __name__ == '__main__':
    model = LangSAM("sam2.1_hiera_small", "../checkpoints/sam2.1/sam2.1_hiera_small.pt")
    image = Image.open("../assets/3.jpg").convert("RGB")
    # text_prompt = "handle"
    # text_prompt = "top small ball"
    # text_prompt = "small black rectangle on the bottom"
    # text_prompt = "small black rectangle on the bottom"
    text_prompt = "road curb"
    results = model.predict([image], [text_prompt])
    print(results)
    show_masks_on_image(image, results[0]['masks'])
