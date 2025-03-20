import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def save_plot(fig: Figure, n):
    folderpath = "./img"
    os.makedirs(folderpath, exist_ok=True)

    if n < 10:
            filename = f'plot_000{n}'
    elif n < 100:
        filename = f'plot_00{n}'
    elif n < 1000:
        filename = f'plot_0{n}'
    else:
        filename = f'plot_{n}'

    save_path = os.path.join(folderpath, filename)
    fig.savefig(save_path, dpi = 200)

def gen_gif():
        """
        Generate a gif from the saved images
        """
        pics = os.listdir('./img')
        gif = []
        for pic in sorted(pics):
            if (pic.endswith('png')):
                gif.append(imageio.imread(f'./img/{pic}'))
        
        gifs = os.listdir('./gif')
        count = 1
        for g in sorted(gifs):
            if (g.endswith('gif')):
                count += 1

        kargs = {'duration' : 100, 'loop' : 12}
        imageio.mimsave(f'./gif/final_{count}.gif', gif, format='GIF', **kargs) # type: ignore