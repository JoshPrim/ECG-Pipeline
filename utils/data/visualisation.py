from PIL import Image, ImageDraw


def visualiseMulti(ecgs, scaling=1):
    """
            Visualises the extracted ECGs in one image per ECG
    :param ecgs: list of ECGs
    :param scaling : a scaling value between 0 and 1 to reduce image size(Details might get lost on smaller sizes)
    """
    if scaling < 0 or scaling > 1:
        raise Exception("Invalid scaling factor ({})".format(scaling))
    for ecg in ecgs:
        # Creates a blank Image that is 30000 pixels high (12 Times 2500 Pixels) and 5000 pixels wide
        imx = Image.new('L', (5000*scaling, 30000*scaling), 255)
        draw = ImageDraw.Draw(imx)
        offset = 28750*scaling
        for lead in ecg:
            # iterating through all leads to generate a single image per ECG, each lead is offset 2500 pixels downwards
            for i in range(len(lead) - 1):
                # Drawing lines between every measuring point
                draw.line((i*scaling, (lead[i] + offset)*scaling, (i + 1)*scaling, (lead[i + 1] + offset)*scaling), fill=0, width=2)
            # reducing the offset so leads dont overlap
            offset = offset - 2500*scaling
        imx.show()


def visualiseIndividual(ecgs):
    """
        Visualises the extracted ECGs in one image per ECG-lead
    :param ecgs: list of ECGs

    """

    for ecg in ecgs:
        for lead in ecg:
            # iterating through all leads to generate a single image per ECG
            # Generating an image for every lead
            ims = Image.new('L', (5000, 2500), 255)
            draw = ImageDraw.Draw(ims)
            for i in range(len(lead) - 1):
                # Drawing lines between every measuring point
                draw.line((i, lead[i], i + 1, lead[i + 1]), fill=0, width=2)
            ims.show()

def visualiseIndividualfromDF(ecg):

    """
        Visualises the extracted ECG in one image per ECG-lead
    :param ecg: ECG in the Format of a pandas Dataframe
    """
    for leadname in ecg.columns:
        # iterating through all leads to generate a single image per ECG
        # Generating an image for every lead
        ims = Image.new('L', (5000, 2500), 255)
        draw = ImageDraw.Draw(ims)
        singlelead = ecg[leadname].tolist()
        for i in range(len(singlelead) - 1):
            # Drawing lines between every measuring point
            draw.line((i, singlelead[i], i + 1, singlelead[i + 1]), fill=0, width=2)
        ims.show()
