from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent.parent
manifest=pd.read_csv(ROOT/'data'/'primary_font_manifest.csv').set_index('family')
families=['Inter','Roboto','Open Sans','Lato','Noto Serif','DejaVu Serif',
          'PT Serif','Liberation Mono','DejaVu Sans Mono','Comic Neue','Lobster Two']
FONT_LABEL='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
image=Image.new('RGB',(1600,1030),'#fbfbfa')
d=ImageDraw.Draw(image)
label=ImageFont.truetype(FONT_LABEL,21)
header=ImageFont.truetype(FONT_BOLD,36)
d.text((60,32),'Одна строка — разные геометрии шрифта',font=header,fill='#24334a')
d.text((60,82),'Открытые реальные шрифты из исследуемого датасета · 128 px/em для измерений',font=label,fill='#677587')
for i, family in enumerate(families):
    row=137+i*79
    if i%2==0:
        d.rectangle((43,row-8,1555,row+68),fill='#f2f6f6')
    item=manifest.loc[family]
    d.text((60,row+16),family,font=label,fill='#344457')
    d.text((325,row+16),item.category,font=label,fill='#78909c')
    typeface=ImageFont.truetype(item.source_file,40)
    d.text((610,row+8),'Hamburgefonts 0123  Wgji',font=typeface,fill='#1f3d47')
image.save(ROOT/'figures'/'font_specimens.png', optimize=True)
print('wrote',ROOT/'figures'/'font_specimens.png')
