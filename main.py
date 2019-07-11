import matplotlib.pyplot as plt
import numpy as np
import dither as dit


photo = dit.open_image('ja.png', size=128)
img_array = np.array(photo)

# new_photo = dit.ordered_dithering(photo, matrix="BAYER", color="BW", palette="WEB", colors=16)

# new_photo = dit.no_dither_color(img_array, palette=dit.grayscale_2bit)

new_photo = dit.error_diffusion(photo, algo="JJN", palette=dit.RGB_3bit, colors=64)

new_photo = dit.upscale(new_photo, 4)

# print(new_photo.convert('RGB').getcolors())

fig = plt.figure(figsize=(18, 9))

ax1 = fig.add_axes([0, 0, 1, 1])
ax1.set_axis_off()
ax1.imshow(new_photo, cmap='gray')
new_photo.save("ccc.png", compress_level=0)
#
fig.canvas.manager.window.wm_geometry("+%d+%d" % (1, 1))
plt.show()
