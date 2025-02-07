import keras
from typing import List
import numpy

class heatmap_creator():
    """ The Grad-CAM heatmap creator class
    """    
    
    
    def __init__(self, model: keras.Sequential, last_conv_layer_name: str, \
        classifier_layer_names: List[str] = ['max_pooling2d_1', 'flatten', 'dropout', 'dense', 'softmax']):
        """ Initial class obtaining the layers from the CNN model

        Args:
            model (keras.Sequential): The model
            last_conv_layer_name (str): The name of the last convolutional layer
            classifier_layer_names (List[str], optional): The list of names of \
                the layers after the last convolutiona layer. \
                    Defaults to ['max_pooling2d_1', 'flatten', 'dropout', 'dense', 'softmax'].
        """
        
        #we create the two models, self.last_conv_layer will provide us the necessary
        #last_conv_feature map  
        last_conv_layer = model.get_layer((last_conv_layer_name))
        self.last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
        
        #the classifier_model input will provide us score
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        layer = classifier_input
        for layer_name in classifier_layer_names:
            layer = model.get_layer(layer_name)(layer)
        self.classifier_model = keras.Model(classifier_input, layer)
    
    def preparing_image(self, img) -> numpy.ndarray:
        """Check if the image is in the right format
        """        

        if type(img) != numpy.ndarray:
            try:
                img = keras.preprocessing.image.img_to_array(img)
            except:
                print("The input should be a numpy.ndarray")
        img = np.expand_dims(img, axis = 0)
        
        return img
    
    def create_heatmap(self, img: numpy.ndarray) -> numpy.ndarray:
        """Creates the heatmap numpy.ndarray. The main implementation of the Grad-CAM algorithm
        """
        #the code computes gradients of the score for feature maps
        #of the last_conv_layer
        with tf.GradientTape() as tape:
            
            last_conv_layer_output = self.last_conv_layer_model(img)
            tape.watch(last_conv_layer_output)
            predictions = self.classifier_model(last_conv_layer_output)
            score_index = tf.argmax(predictions[0])
            scores = predictions[:, score_index]
            
        gradients = tape.gradient(scores, last_conv_layer_output)
        weights = tf.reduce_mean(gradients, axis = (0,1,2))
        
        
        #the implementation of the second equation
        #the feature maps are multiplied by the weights
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        weights = weights.numpy()
        for i in range(weights.shape[-1]):
            last_conv_layer_output[:,:,i] *= weights[i]
            
            
        #finally we create a heatmap, using the neuron activity map
        #by normalizing it and replacing negative places for zero
        heatmap = np.mean(last_conv_layer_output, axis = -1)
        heatmap = np.maximum(heatmap, 0)/np.max(heatmap)
        
        return heatmap
    
    def create_visualization(self, img) -> None:
        '''Create the visualization for the image and its heatmap
        '''
        
        #preparing the image and heatmap
        img = self.preparing_image(img)
        heatmap = self.create_heatmap(img)
        img = np.squeeze(img, axis = 0)
        
        heatmap = np.uint8(255 * heatmap)
        img = np.uint8(255 * img)
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        #rescalling the image
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        
        #merging the original image with the heatmap
        sup_img = jet_heatmap * 0.5 + img
        sup_img = keras.preprocessing.image.array_to_img(sup_img)
        
        plt.matshow(sup_img)
        plt.matshow(img, cmap = 'binary')
        plt.matshow(heatmap)
        plt.show()