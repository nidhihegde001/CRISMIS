import requests
import re
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from lacosmic import lacosmic
from planetaryimage import PDS3Image
from PIL import Image


def preprocess_image(directory, name):
    url = 'https://pdsimage2.wr.usgs.gov/archive/mess-e_v_h-mdis-2-edr-rawdata-v1.0/MSGRMDS_1001/DATA/'+ str(directory)
    file = requests.get(url+ '/' + str(name))
    try: 
        img = open("img.txt", "wb+")
        img.write(file.content)
        image = PDS3Image.open('img.txt')
        t = image.image
        plt.imsave('images/'+ name +'.jpg', t, cmap = 'gray')

    except:
        img_data = bytearray()
        header = []
    #   print(file.content)

        # Test takes 2 values:
        # 0-> Before end of header is reached
        # 1-> End of header and start of image data
        test = 0
        for line in file.iter_lines():
            # Read Image data
            if test: 
                img_data.extend(line)

            # End of header is reached
            elif re.match("^END$", line.decode('utf-8').strip()):
                test = 1

            # Read the Header
            else:
                header.append(line.decode('utf-8').strip())

        file.close()
    #         print(header)
        # Flag takes 3 values:
        # 0-> before required data is encountered
        # 1-> Reading required data
        # 2-> after the required data is read
        flag = 0
        for line in header:
            
            if re.match('^TARGET_NAME',line):
                target = line.split('"')[1]
                # print(target)

            # Start Byte
            if re.match('^\^IMAGE',line):
                n = line.split('=')
                start_byte = int(n[1].strip())

            # End of required data 
            if re.match('^END_OBJECT',line):
                flag = 2

            # Beginning of required data
            if re.match('^OBJECT',line):
                flag = 1

            # Required data
            if flag == 1:
                if re.match('^LINES',line):
                    # Extract no of rows
                    a = line.split()
                    row = int(a[2])

                elif re.match('^LINE_SAMPLES',line):
                    # Extract no of columns
                    a = line.split()
                    column = int(a[2])

                elif re.match('^SAMPLE_BITS',line):
                    # Extract the type of bit encoding
                    a = line.split()
                    typ = 'uint'+ a[2]

        # print(row,column,typ)
        #     print(len(img_data.decode()))

        # b => bit width of data
        b = int(a[2])

        # Numpy array from byte array
        a = np.frombuffer(img_data[len(img_data)%16:],dtype=typ)
        #     print(a)

        # s => No of extra bits in the image data = Total no of bits - rows*columns
        s = (a.size/row -column)*row

        if s<0:
            print("Defective Image:",s)
            return None

        # t => Array after Trash data is removed
        t = a[int(s):].copy()
        # print(t.size)
        t.resize((row,column))

        # Reshaping the array
        np.divide(t,float(2**(b-8)))
        t = t.astype('uint8')

    # Save image
        plt.imsave('images/'+ name +'.jpg',t,cmap='gray')
#     plt.show()

    image = Image.open('images/' + name +'.jpg').convert("L")
    data = np.array(image)
    # print(data.shape)
    crmask,n_cosmic = lacosmic(data, contrast=5.0, cr_threshold=4.5, neighbor_threshold=0.3, 
                                    error=None, mask=None, background=None, effective_gain=1.0,
                                    readnoise=6.5, maxiter=1, border_mode=u'mirror')
    if (n_cosmic > 0):
        return data
    else:
        print("Cosmic Ray not found")
