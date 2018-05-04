from PIL import Image
import glob, os
import sys
import scipy.misc
size = (64, 64)

rootdir = "D:\Wikiart\wiki"
ext = ".jpg"
for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        name = subdir+"\\"+f
        im = Image.open(name)
        original_width, original_height = im.size
        crop_width = int(original_width * 0.9)
        crop_height = int(original_height * 0.9)
        diff_width = original_width - crop_width
        diff_height = original_height - crop_height
        # (x1, y1, 0 -> w
        #                  0
        #                  |
        #                  h  
        #           
        #        x2, y2)
        # Left: The x-coordinate of the leftmost edge of the box.

        # Top: The y-coordinate of the top edge of the box.

        # Right: The x-coordinate of one pixel to the right of the rightmost edge of the box. This integer must be greater than the left integer.

        # Bottom: The y-coordinate of one pixel lower than the bottom edge of the box. This integer must be greater than the top integer. 
        try:
            # Resize original 
            try:
                im_resized = im.resize(size, Image.ANTIALIAS)
                im_resized.save(name, "JPEG")
            except IOError:
                print("error - couldn't resize ",name)

            # Top left crop and resize
            top_left = (0,0,crop_width,crop_height) 
            tl = im.crop(top_left)
            try:
                tl = tl.resize(size, Image.ANTIALIAS)
                tl.save(name[:-4]+"_top_left"+".jpg")    
            except IOError:
                print("error - couldn't resize",name)

            # Top right crop and resize
            top_right = (diff_width,0,original_width,crop_height)
            tr = im.crop(top_right)
            try:
                tr = tr.resize(size, Image.ANTIALIAS)
                tr.save(name[:-4]+"_top_right"+".jpg")    
            except IOError:
                print("error - couldn't resize",name)

            # Center crop and resize
            # with thanks to https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
            right = (original_width + crop_width) // 2
            bottom = (original_height + crop_height) // 2
            center = (diff_width//2,diff_height//2,right,bottom)
            cr = im.crop(center)
            try:
                cr = cr.resize(size, Image.ANTIALIAS)
                cr.save(name[:-4]+"_center"+".jpg")    
            except IOError:
                print("error - couldn't resize",name)

            # Bottom left crop and resize
            bottom_left = (0,diff_height,crop_width,original_height)
            bl = im.crop(bottom_left)
            try:
                bl = bl.resize(size, Image.ANTIALIAS)
                bl.save(name[:-4]+"_bottom_left"+".jpg")   
            except IOError:
                print("error - couldn't resize",name)

            # Bottom right crop and resize
            bottom_right = (diff_width,diff_height,original_width,original_height)
            br = im.crop(bottom_right)
            try:
                br = br.resize(size, Image.ANTIALIAS)
                br.save(name[:-4]+"_bottom_right"+".jpg")   
            except IOError:
                print("error - couldn't resize",name)
        except IOError:
            print("Couldn't do ",name)
