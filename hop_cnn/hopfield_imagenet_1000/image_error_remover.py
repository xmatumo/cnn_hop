# Delete unreadable files from the image directory.
# Once you do it is okay.

# import library

from PIL import Image
import os


# remove error
class Image_Error_Remover:

    def remove(self, data_dir):
        
        # load target directory 
        data_dir = data_dir
        dir_name = [i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))]
        
        # if image is error, remove the image
        for dirs in dir_name:
            file_name = [i for i in os.listdir(os.path.join(data_dir, dirs))]
            for files in file_name:
                target_file = os.path.join(data_dir, dirs, files)
                try:
                    test = Image.open(target_file)
                except:
                    print("remove:"+target_file)
                    os.remove(target_file) 
                else:
                    test.close()
