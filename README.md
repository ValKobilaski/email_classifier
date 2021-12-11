# email_classifier
#PreRequisits:
    1. Ensure all necessary libraries are installed
        pip install -r requirements.txt
    2. Ensure Data is accessible
        /CSDMC2010_SPAM/TRAINING exists from the cwd
    3. Ensure Spam Words list exists
        filter_words.txt in cwd
    4. Ensure training labels exists
        /CSDMC2010_SPAM/SPAMTrain.label
        
#Execution
    1. Run Precprocessing
        python preprocess_emails.py
    2. Run model
        python model.py