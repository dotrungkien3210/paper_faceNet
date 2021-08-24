from library.preprocess import preprocesses

input_data_directory  = './raw_imagesTest'   ## Directory of the raw images. These images must have high variance.
output_data_directory = './face_datasetTest' ## Directory to save the faces from the images
## Creating the object by specifying source directory and Output directory which is "Face Dataset" to where the data is stored
filter_obj = preprocesses(input_data_directory, output_data_directory)

## Starting to collect the detected faces from images within the input directory 'raw_images' and
### saving the detected faces in the 'face_dataset'
no_of_total_images, _ = filter_obj.collect_data()
print('Total raw images found from the input directory= %d' % no_of_total_images)