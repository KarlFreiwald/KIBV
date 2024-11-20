import torch
import torchvision

from config import num_classes


class FCN32s(torch.nn.Module):
    def __init__(self, num_classes=num_classes):
        super(FCN32s, self).__init__()

        # Lade das vortrainierte VGG16-Modell und entferne den Klassifikator
        self.vgg16 = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1,
            progress=True
        )
        self.features = self.vgg16.features  # Die Faltungsschichten bleiben unverändert

        # Entfernen der voll verbundenen Schichten und ersetzen durch Faltungen
        self.conv1x1 = torch.nn.Conv2d(
            512,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Transponierte Faltung (Deconvolution) zum Upsampling der Ausgaben
        self.deconv = torch.nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)

    def forward(self, x):
        # Die Faltungsschichten von VGG16 durchlaufen
        x = self.features(x)

        # 1x1 Faltung, um die Klassifikationsvorhersagen für jedes Pixel zu berechnen
        x = self.conv1x1(x)

        # Deconvolution (Upsampling) der groben Ausgaben auf eine feinere Auflösung
        x = self.deconv(x)

        return x
