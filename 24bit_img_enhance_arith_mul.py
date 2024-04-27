# import cv2
# import numpy as np

# # Load the RGB image
# image_path = '24bit_rgb.jpeg'  # Update with the actual path to your image
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Image could not be loaded. Please check the file path.")
#     exit()

# # Specify the color channel to enhance (0 for Blue, 1 for Green, 2 for Red)
# channel_to_enhance = 2  # Enhance the Red channel

# # Enhancement factor (adjust as needed)
# enhancement_factor = 1.5  # Example: increase the Red channel by 50%

# # Split the image into its RGB channels
# b, g, r = cv2.split(image)

# # Apply enhancement to the selected channel
# enhanced_channel = r if channel_to_enhance == 2 else g if channel_to_enhance == 1 else b
# enhanced_channel = np.clip(enhanced_channel * enhancement_factor, 0, 255).astype(np.uint8)

# # Merge the enhanced channel back into the image
# enhanced_image = cv2.merge([b, g, enhanced_channel]) if channel_to_enhance == 0 else \
#                  cv2.merge([b, enhanced_channel, r]) if channel_to_enhance == 1 else \
#                  cv2.merge([enhanced_channel, g, r])

# # Display the enhanced image
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
import ctypes



def get_screen_dimensions():
    user32=ctypes.windll.user32
    width, height= user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return width ,height


image=cv2.imread('ippr_assignment_1a_img1.jpg')

channel_to_enhance= 2

enhancement_factor=1.5

b,r,g= cv2.split(image)

enhanced_channel=r if channel_to_enhance==2 else g if channel_to_enhance==1 else b
enhanced_channel= np.clip(enhanced_channel*enhancement_factor, 0, 255).astype(np.uint8)

enhanced_image=cv2.merge([b,g,enhanced_channel])if channel_to_enhance == 0 else \
              cv2.merge([b,enhanced_channel, r]) if channel_to_enhance==1 else \
              cv2.merge([enhanced_channel,g,r])

screen_width, screen_height= get_screen_dimensions()

scaled_image=cv2.resize(enhanced_image, (screen_width-100, screen_height-100))

cv2.imshow('Enhanced Image', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
