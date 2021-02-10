from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def visualiseMulti(ecgs, scaling=1):
    """
            Visualises the extracted ECGs in one image per ECG
    :param ecgs: list of ECGs
    :param scaling : a scaling value between 0 and 1 to reduce image size(Details might get lost on smaller sizes)
    """
    if scaling < 0 or scaling > 1:
        scaling = 1
    for ecgid in ecgs:
        ecg = ecgs[ecgid]
        # Creates a blank Image that is 30000 pixels high (12 Times 2500 Pixels) and 5000 pixels wide
        imx = Image.new('L', (int(5000 * scaling), int(30000 * scaling)), 255)
        draw = ImageDraw.Draw(imx)
        offset = 1250
        for leadid in ecg['leads']:
            lead = ecg['leads'][leadid]
            # iterating through all leads to generate a single image per ECG, each lead is offset 2500 pixels downwards
            for i in range(len(lead) - 1):
                # Drawing lines between every measuring point
                draw.line(
                    (i * scaling, (-lead[i] + offset) * scaling, (i + 1) * scaling, (-lead[i + 1] + offset) * scaling),
                    fill=0, width=2)
            # reducing the offset so leads dont overlap
            offset = offset + 2500
        imx.show()


def visualiseIndividualfromDF(ecg, scaling=1):
    """
        Visualises the extracted ECG in one image per ECG-lead
    :param scaling: a scaling value between 0 and 1 to reduce image size(Details might get lost on smaller sizes)
    :param ecg: ECG in the Format of a pandas Dataframe
    """
    if scaling < 0 or scaling > 1:
        scaling = 1
    for leadname in ecg.columns:
        # iterating through all leads to generate a single image per ECG
        # Generating an image for every lead
        ims = Image.new('L', (int(5000 * scaling), int(2500 * scaling)), 255)
        draw = ImageDraw.Draw(ims)
        singlelead = ecg[leadname].tolist()
        draw.text((10, (2300 * scaling)), leadname, font=ImageFont.load_default())
        for i in range(len(singlelead) - 1):
            # Drawing lines between every measuring point
            draw.line((i * scaling, (-singlelead[i] + 1250) * scaling, (i + 1) * scaling,
                       (-singlelead[i + 1] + 1250) * scaling)
                      , fill=0, width=2)
        ims.show()


def visualiseIndividualinMPL(ecg):
    """
        Visualises the extracted ECG in one Plot per ECG-lead. This Method uses Matplotlib.
    :param ecg: ECG in the Format of a pandas Dataframe
    """
    for leadname in ecg.columns:
        # iterating through all leads to generate a single image per ECG
        # Generating an image for every lead
        ecg.plot(kind='line', y=[leadname])
        plt.show()
