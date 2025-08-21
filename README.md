# Face Recognition APIs

This implementation is based on the article [Hailo guide: Comprehensive guide to building a face recognition system](https://community.degirum.com/t/hailo-guide-comprehensive-guide-to-building-a-face-recognition-system/143)

The above guide from DeGirum Team explains how to build a face database, and how to recognize faces in a image / video using the vector database and machine learning models optimized for Hailo processors. The models are avilable at Degirum's model zoo.


## Approach
Register face embedding (512 sized vector) for every known face with a label/id. when a new image is given, first detect faces in it using face detection model, then for each face align and find embedding and search the database for similar faces. Identify the face if the similarity score is above threshold. 


## Scope
The scope of this project is to get a more robust API out of this, with the purpose of gradually building a face database for photo collections on your Raspberry Pi 5 + Hailo processor. (currently usign Hailo-8).

The hardware interface must be clearly abstracted out so that we can change the model or hardware in later stage.

## Database
LanceDB is enought for similarity search, however, it can't be used alone due to various limitation in managing non-vector data. Hence we use SQLite based SQL database to store all other informations. like addressbook, associating multiple faces to same person. The images of the faces are stored as png files in a folder. 

## Models
    This project uses the following models downloaded from Degirum's Model zoo.
    1. retinaface_mobilenet--736x1280_quant_hailort_hailo8_1 - for face detection
    2. arcface_mobilefacenet--112x112_quant_hailort_hailo8_1 - for face embedding


## Requirements
         * A face can be registred by sending a face image and the person in it at anytime. 
            if the id of the person is provided, it will be attached with that person
            if name of the person is provided, a new person is automatically registered.
        * A face is always assicated with a person
        * A face may be moved from one person to another person as a correction
        * A face can be deleted to avoid getting detected in future

        * Multiple persons can have same name, hence an unique id is used
        * A person is associated with atleast one face. 
        * A person already exists on the store is identified by id, not name
        * If a person is deleted, all the faces assocciated with that person is deleted
        * If the last known person is deleted, the person is automatically deleted.

        * For the image provided, 
            * All the faces in that image is detected
            * There are two threshold
                1. if confidence scopre is above threshold1, a person id and confidence score will be returned.
                2. if none above threshold1, threshold2 will be considerred. If more than one person is found in this score all will be returned with confidence score. 
                User can use this information to manually check and register new face.
            * For the unknown persons, only the containing box will be returned
                User can crop, name and then register that person.

