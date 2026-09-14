"""Contact sheets include every planned view; gallery indices do not use image scores."""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]/'outputs/mobility_repair_campaign_20260914_v3/native'


def main():
    representatives = []
    for case in sorted(ROOT.glob('*')):
        if not (case/'view_plan.json').is_file():
            continue
        count = json.loads((case/'view_plan.json').read_text())['view_count']
        for engine, folder in [('Unity', 'unity_project'), ('Unreal', 'unreal_run_01')]:
            images = {int(re.fullmatch(r'interior_(\d+)\.png', path.name)[1]): path
                      for path in (case/folder).glob('interior_*.png')
                      if re.fullmatch(r'interior_(\d+)\.png', path.name)}
            if sorted(images) != list(range(1, count+1)):
                continue
            width, height, columns = 240, 152, 6
            sheet = Image.new('RGB', (columns*width, ((count+columns-1)//columns)*height), '#20252a')
            draw = ImageDraw.Draw(sheet)
            for number, path in sorted(images.items()):
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    image.thumbnail((width-4, height-20))
                    x, y = ((number-1) % columns)*width, ((number-1)//columns)*height
                    sheet.paste(image, (x+(width-image.width)//2, y))
                    draw.text((x+4, y+height-17), f'View {number}', fill='white')
            sheet.save(case/f'{folder}_contact.jpg', quality=90)
            # Stable middle floor-direction view, independent of its appearance.
            selected = 1+2*((count//2)//2)
            representatives.append((f'{case.name} · {engine} · view {selected}', images[selected]))
    if len(representatives) == 12:
        width, height = 520, 324
        gallery = Image.new('RGB', (width*2, height*6), '#20252a')
        draw = ImageDraw.Draw(gallery)
        for i, (label, path) in enumerate(representatives):
            with Image.open(path) as image:
                image = image.convert('RGB')
                image.thumbnail((width-8, height-25))
                x, y = (i % 2)*width, (i//2)*height
                gallery.paste(image, (x+(width-image.width)//2, y))
                draw.text((x+5, y+height-20), label, fill='white')
        gallery.save(EVIDENCE/'native_gallery.jpg', quality=92)
    print(f'{len(representatives)}/12 complete engine contact sheets')


if __name__ == '__main__':
    main()
