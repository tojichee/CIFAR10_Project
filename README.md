Step 1: Get into the Correct Folder Using the Command Line
First, you had to tell the computer where your training script was located. You did this by opening a command line terminal (like PowerShell or Command Prompt) and navigating to the right directory.
You used the "address bar trick" to do this easily:
You opened your training folder in the Windows File Explorer.
You clicked the address bar, typed cmd or powershell, and pressed Enter.
This opened a terminal window that was already in the correct folder:
C:\Users\Brendan\Downloads\Applied_AI_Final_Project\training
Step 2: Modify the Script to Save Correctly
You identified that TensorFlow was saving the model as a folder instead of a single .h5 file. You fixed this with a crucial one-line code change in your train_cifar10_model.py script.
You found the line:
Generated python
model.save(model_save_path)
Use code with caution.
Python
And you changed it to explicitly force the H5 format:
Generated python
model.save(model_save_path, save_format='h5')
Use code with caution.
Python
Step 3: Run the Python Script from the Command Line
This was the final, critical action. With the terminal open in the correct folder and the script modified, you executed the command that started the entire process:
Generated powershell
python train_cifar10_model.py
Use code with caution.
Powershell
This command told the Python 3.11 interpreter to:
Open and run your train_cifar10_model.py script.
Build the CNN architecture.
Load the CIFAR-10 dataset.
Train the model for 10 epochs.
Evaluate its accuracy.
Finally, save the fully trained model into a single, complete cifar10_cnn_model.h5 file in that same training folder.
In short, you ran the command python train_cifar10_model.py in the terminal after navigating to the correct folder and modifying the save command. That's what produced your final, working .h5 file.
18.8s
