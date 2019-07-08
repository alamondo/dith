from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dither as dit

f = Image.open("garden.png")
f = f.convert("RGB")
a = f.copy()

size = 1024

f.thumbnail((size, size), Image.ANTIALIAS)
f = np.array(f)
f = dit.ordered_dithering_color(f, dit.c64_palette)

fig = plt.figure(figsize=(18, 9))

ax1 = fig.add_axes([0, 0, 1, 1])
ax1.set_axis_off()
ax1.imshow(f, cmap='gray')

im = Image.fromarray(np.uint8(f))
im.save("bbb.png", compress_level=9)

fig.canvas.manager.window.wm_geometry("+%d+%d" % (1, 1))
plt.show()
