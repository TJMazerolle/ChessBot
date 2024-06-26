model.h5 - Model trained on only the initial random board states

new_model.h5, old_model.h5 - Models trained by reinforcement learning. new_model.h5 and old_model.h5 were made to play against each other.

ChessFunctions.py - Just a group of helper functions used in the rest of the scripts.

GeneratingTrainingStatesTroy2.ipynb - This script generates the random board states and saves them.  In order to avoid memory crashes it generates 1000 board states then saves them.

MergeTensors.ipynb - This script takes all the board states generated by GeneratingTrainingStatesTroy2.ipynb, merges them into one tensor, and saves the result.

TrainingCode.ipynb - This script takes the output from MergeTensors.ipynb and trains the initial model to produce model.h5.  

ReinforcementTraining.ipynb - This script takes new_model.h5 and old_model.h5, makes them play against each other, and updates the models for continued learning.  Note that the script originally loaded model.h5 when first starting training, but now loads the other two models since they are the most current versions of the models that had reinforcement learning done to them.

Implementation.ipynb - This script loads model.h5 and new_model.h5 and pits them against Stockfish to evaluate how each model does.

PlayAgainstBot.ipynb - This script is not important for this project, but it allows us to play against the bot ourselves when we want to try that.