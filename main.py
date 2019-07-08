from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dither as dit

photo = Image.open("garden.png")
photo = photo.convert("RGB")

size = 512

photo.thumbnail((size, size), Image.ANTIALIAS)
# photo = photo.convert("L")

photo = dit.ordered_dithering(photo, matrix="BAYER", color="RGB", palette=dit.c64_palette)

fig = plt.figure(figsize=(18, 9))

ax1 = fig.add_axes([0, 0, 1, 1])
ax1.set_axis_off()
ax1.imshow(photo, cmap='gray')

im = Image.fromarray(np.uint8(photo))
im.save("bbb.png", compress_level=9)

fig.canvas.manager.window.wm_geometry("+%d+%d" % (1, 1))
plt.show()
