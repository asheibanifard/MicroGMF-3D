import tifffile
with tifffile.TiffFile('data/raw/10-2900-control-cell-05_cropped_corrected.tif') as tif:
    for page in tif.pages:
        print("Shape:", page.shape)
        print("Metadata:", page.tags)
        # Check for pixel resolution info
        if 'XResolution' in page.tags:
            print("XRes:", page.tags['XResolution'].value)
        if 'YResolution' in page.tags:
            print("YRes:", page.tags['YResolution'].value)
        break
