import numpy

import PIL
import selectivesearch
from neural import SignsNeural, SeekNeural

if __name__ == "__main__":
    signs_neural = SignsNeural()
    signs_neural.prepare()

    eq_img = PIL.Image.open('equations.png')
    eq_img_data = numpy.asarray(eq_img)

    _, regions = selectivesearch.selective_search(eq_img_data, sigma=0.9, min_size=10)
    for region in regions:
        rect = (
            region['rect'][0],
            region['rect'][1],
            region['rect'][0] + region['rect'][2],
            region['rect'][1] + region['rect'][3],
        )
        sign_img = eq_img.crop(rect)
        predicts = signs_neural.get_sign(sign_img)
        print(rect, predicts)
        sign_img.show()
        input('>')

    print('\b')
