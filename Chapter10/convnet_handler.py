from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier

class ConvNetClassifier(ImageClassifier):
    """
    Extends the ImageClassifier handler meant to handle image data for image classification. We have adpated it to convert color (RGB) images to grayscale images resized to 28x28 pixels as well as normalizing pixel values. The postprocess method ensures to return the digit with the highest prediction probability.
    """

    image_processing = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1302,), (0.3069,))
    ])

    def postprocess(self, output):
        return output.argmax(1).tolist()